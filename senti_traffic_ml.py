"""
senti_traffic_ml.py
===================
Senti Traffic -- the ML layer, and ONLY the ML layer.

This is the single machine-learning component of the whole system: an object
detector that looks at a frame and says "there is a motorcycle here, a person
there, a car over there."

Everything else in Senti Traffic -- tracking, the rule engine, the rolling
evidence buffer, the admin portal, challan generation -- is ordinary Python
built on top of what this file returns. Nothing downstream imports a model.

WHAT THIS FILE DOES NOT DO (by design):
  - decide whether something is a violation   -> rule engine
  - remember anything between frames          -> tracker
  - read number plates                        -> separate OCR stage
  - talk to a database or a portal            -> backend

You do not train anything here -- see senti_traffic_train.py for that. Out of
the box the weights are pretrained on COCO and downloaded on first run; point
--weights at a fine-tuned checkpoint and this file picks up the new classes
(auto_rickshaw included) automatically, because the taxonomy is read off the
model rather than hard-coded.

SETUP
-----
    pip install ultralytics opencv-python

RUN
---
    python senti_traffic_ml.py --source path/to/traffic.mp4
    python senti_traffic_ml.py --source path/to/traffic.mp4 --show
    python senti_traffic_ml.py --source 0                      # webcam
"""

from __future__ import annotations

import argparse
import contextlib
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import cv2

try:
    import torch
except ImportError:  # torch ships with ultralytics, but stay graceful
    torch = None

from ultralytics import YOLO

# ultralytics 8.4 renamed the FP16 flag: `half=True` is now `quantize=16`, and
# the deprecation shim warns whenever `half` is passed AT ALL -- including
# half=False. Passing it per frame therefore prints a warning per frame, which
# buries real errors in a live deployment. Detect which spelling this install
# wants, once, and use it.
try:
    from ultralytics.cfg import DEFAULT_CFG_DICT as _ULTRA_CFG
except Exception:  # noqa: BLE001 -- private-ish path; degrade to the old flag
    _ULTRA_CFG = {}

SUPPORTS_QUANTIZE = "quantize" in _ULTRA_CFG


# ---------------------------------------------------------------------------
# 1. TAXONOMY
# ---------------------------------------------------------------------------
# The vocabulary the rule engine speaks. This list -- not COCO, not any
# particular checkpoint -- is the contract. `senti_traffic_train.py` writes the
# same list into data.yaml, so a fine-tuned model and this file cannot drift.
#
# ORDER MATTERS for training: a YOLO dataset's class IDs are positions in this
# list. Append new classes at the end; never reorder, or every label file ever
# written points at the wrong class.
SENTI_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "auto_rickshaw",
)

# Pretrained COCO IDs, for the stock yolo11*.pt path only.
#
# This used to be the sole source of truth, which was a latent bug: these are
# COCO's IDs, and a fine-tuned model numbers its classes by its own dataset
# order. Point the old code at Indian weights and class 2 stops meaning "car"
# while nothing complains. The detector now reads names off the loaded model
# (see _build_taxonomy) and this table is only the fallback for checkpoints
# that carry no usable .names.
COCO_TO_TRAFFIC: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Whatever a checkpoint happens to call these, they mean the same thing to us.
# Ultralytics COCO says "motorbike" in some exports; annotation tools and public
# Indian datasets spell three-wheelers a dozen ways. Names are lowercased and
# stripped of spaces/hyphens before lookup.
CLASS_ALIASES: dict[str, str] = {
    "motorbike": "motorcycle",
    "bike": "motorcycle",
    "scooter": "motorcycle",
    "cycle": "bicycle",
    "pedestrian": "person",
    "auto": "auto_rickshaw",
    "autorickshaw": "auto_rickshaw",
    "rickshaw": "auto_rickshaw",
    "tuktuk": "auto_rickshaw",
    "threewheeler": "auto_rickshaw",
    "three_wheeler": "auto_rickshaw",
    "minibus": "bus",
    "lorry": "truck",
}
# Deliberately NOT aliased, because the right answer depends on the deployment
# and a silent default would be wrong half the time:
#   van    -- passenger van (car) or light goods vehicle (truck)?
#   rider  -- a person on a two-wheeler; mapping it to `person` is usually
#             right for helmet/triple-riding work, but dropping it teaches the
#             model that riders are background, which is actively harmful.
#   tractor / cart / construction vehicle -- vehicles, but not ones any current
#             rule reasons about.
# Pass them explicitly: senti_traffic_train.py remap --map "Rider=person"

# Classes that carry a rider exposed to the elements. The helmet and
# triple-riding rules only ever look at these, so we can skip everything else
# and save a lot of work downstream.
TWO_WHEELER_CLASSES = {"motorcycle", "bicycle"}

# Anything that counts as a vehicle for violation purposes.
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck", "auto_rickshaw"}

# INDIAN CONDITIONS -- now trainable, not just documented
# -------------------------------------------------------
# COCO has no auto-rickshaw class, and the pretrained model scatters
# three-wheelers across car / truck / motorcycle depending on the angle. That
# is a property of the *weights*, not of this code: `auto_rickshaw` is a first
# class citizen of SENTI_CLASSES above and appears the moment you load a
# checkpoint that was trained on it.
#
# TrafficDetector.trained_classes tells you which of SENTI_CLASSES the loaded
# weights can actually produce, and .missing_classes tells you what is absent,
# so the rule engine can refuse to run a three-wheeler rule on COCO weights
# instead of silently scoring it against detections that will never arrive.
COCO_BLIND_SPOTS = {"auto_rickshaw"}


# Per-class confidence thresholds.
#
# Why not one global threshold? Because dense Indian traffic detects unevenly.
# Motorcycles are small, frequently overlapping, and partially occluded, so a
# strict threshold silently drops the vehicles we most care about. Cars are
# large and unambiguous, so we can afford to be strict and cut false positives.
#
# These are starting values -- tune them against your own footage.
CLASS_CONF_THRESHOLDS: dict[str, float] = {
    "person": 0.35,
    "bicycle": 0.30,
    "motorcycle": 0.30,
    "car": 0.45,
    "bus": 0.45,
    "truck": 0.45,
    # Freshly fine-tuned and trained on far less data than COCO's classes.
    # Start permissive, then raise it once you have a validation run to read.
    "auto_rickshaw": 0.30,
}

DEFAULT_CONF = 0.35

# The coarse floor handed to YOLO itself. Detections below this are dropped
# inside the model, before _parse ever sees them, so the floor MUST sit at or
# below every per-class threshold *and* below DEFAULT_CONF.
#
# Taking min() over CLASS_CONF_THRESHOLDS alone was wrong: a class added to
# SENTI_CLASSES without its own entry falls back to DEFAULT_CONF, and if every
# listed threshold happened to be stricter than DEFAULT_CONF the model would
# silently cut that class off above its own threshold. Include DEFAULT_CONF and
# the fallback path stays honest no matter how the tables are edited.
CONF_FLOOR = min([*CLASS_CONF_THRESHOLDS.values(), DEFAULT_CONF])


def canonical_class(name: str) -> Optional[str]:
    """Map whatever a checkpoint calls a class onto our vocabulary.

    Returns None for anything outside SENTI_CLASSES -- traffic lights, dogs,
    and the other 70-odd COCO classes we do not model.
    """
    key = str(name).strip().lower().replace(" ", "").replace("-", "")
    key = CLASS_ALIASES.get(key, key)
    if key in SENTI_CLASSES:
        return key
    # aliases may map to an underscored canonical name that survived stripping
    key = key.replace("_", "")
    for known in SENTI_CLASSES:
        if known.replace("_", "") == key:
            return known
    return None


def build_taxonomy(model_names) -> dict[int, str]:
    """Derive {model class id -> our name} from a checkpoint's own label list.

    `model_names` is ultralytics' `model.names`: {id: label} for whatever the
    weights were trained on. Reading it here is what lets one file serve both
    stock COCO weights and Indian fine-tunes -- the IDs differ, the names do
    not.
    """
    if not model_names:
        return dict(COCO_TO_TRAFFIC)

    items = model_names.items() if isinstance(model_names, dict) else enumerate(model_names)
    taxonomy = {}
    for cls_id, label in items:
        name = canonical_class(label)
        if name is not None:
            taxonomy[int(cls_id)] = name
    return taxonomy or dict(COCO_TO_TRAFFIC)


# ---------------------------------------------------------------------------
# 2. OUTPUT CONTRACT
# ---------------------------------------------------------------------------
# This dataclass is the boundary between "ML" and "everything else". The rule
# engine consumes lists of these and never touches a tensor, a model, or a
# YOLO object. Keeping this contract stable means the detector can be swapped
# (different model, different framework, even a hardware sensor) without a
# single downstream file changing.


@dataclass
class Detection:
    """One detected object in one frame."""

    cls_name: str                      # "motorcycle", "person", ...
    confidence: float                  # 0.0 - 1.0
    xyxy: tuple[float, float, float, float]   # box: (x1, y1, x2, y2) in pixels
    track_id: Optional[int] = None     # filled in only when tracking is on
    frame_index: int = -1

    # -- convenience geometry, used constantly by the rule engine ------------

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Where the object meets the road.

        This -- not the box centre -- is the point rules should test against
        stop lines and lane polygons. A tall truck's centre floats metres above
        the tarmac and will cross a painted line long before its wheels do.
        """
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_vehicle(self) -> bool:
        return self.cls_name in VEHICLE_CLASSES

    @property
    def is_two_wheeler(self) -> bool:
        return self.cls_name in TWO_WHEELER_CLASSES


@dataclass
class FrameResult:
    """Everything the ML layer knows about a single frame."""

    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

    # When this frame happened, in milliseconds from the start of the stream.
    #
    # frame_index alone cannot answer "was the light red 1.8 seconds ago?" --
    # that needs elapsed time, and dividing the index by a nominal FPS is wrong
    # the moment a frame is dropped or the camera runs off-nominal. run_stream
    # fills this from the container's own clock where the source has one, and
    # falls back to wall-clock for live feeds. -1.0 means "not supplied".
    timestamp_ms: float = -1.0

    # Nominal source FPS, or -1.0 if the source will not say. Speed and
    # red-light-duration rules need it; read it from here rather than assuming.
    source_fps: float = -1.0

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ms / 1000.0 if self.timestamp_ms >= 0 else -1.0

    def of_class(self, *names: str) -> list[Detection]:
        wanted = set(names)
        return [d for d in self.detections if d.cls_name in wanted]

    @property
    def vehicles(self) -> list[Detection]:
        return [d for d in self.detections if d.is_vehicle]

    @property
    def two_wheelers(self) -> list[Detection]:
        return [d for d in self.detections if d.is_two_wheeler]

    @property
    def persons(self) -> list[Detection]:
        return self.of_class("person")


# ---------------------------------------------------------------------------
# 3. THE DETECTOR
# ---------------------------------------------------------------------------


class TrafficDetector:
    """Pretrained YOLO wrapped in a stable, traffic-specific interface.

    The entire ML surface area of Senti Traffic lives inside this class.
    """

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        device: Optional[str] = None,
        imgsz: int = 640,
        min_box_area: float = 400.0,
        half: Optional[bool] = None,
    ) -> None:
        """
        weights      Model file. Downloaded automatically if absent.
                     yolo11n = nano  (fastest, weakest -- good for laptops)
                     yolo11s / m / l = progressively slower and more accurate.
        device       "cuda", "cpu", or None to auto-detect.
        imgsz        Inference resolution. Larger sees small/distant objects
                     better but costs time. 640 is the usual balance; go to
                     960+ if plates and distant two-wheelers are being missed.
        min_box_area Reject boxes smaller than this (px^2). A 12x20px blob far
                     down the road is not usable as evidence even if it is
                     genuinely a vehicle -- discarding it early keeps the rule
                     engine clean.
        half         FP16 inference. Roughly 2x faster on GPU, no meaningful
                     accuracy loss. Forced off on anything but CUDA.
        """
        self.device = self._normalise_device(device or self._auto_device())
        self.imgsz = imgsz
        self.min_box_area = min_box_area

        # FP16 is a CUDA-only trick, and not every torch build politely ignores
        # it elsewhere -- a half-precision forward pass on CPU dies with
        # "slow_conv2d_cpu not implemented for 'Half'". Clamp it here so the
        # flag can never reach a non-CUDA device, however it was passed in.
        on_cuda = self.device.startswith("cuda")
        self.half = on_cuda if half is None else bool(half) and on_cuda

        # FP32 is the default in every ultralytics version, so say nothing at
        # all when we are not asking for FP16 -- naming the flag is what
        # triggers the deprecation warning, regardless of its value.
        if not self.half:
            self._precision_kwargs: dict = {}
        elif SUPPORTS_QUANTIZE:
            self._precision_kwargs = {"quantize": 16}
        else:
            self._precision_kwargs = {"half": True}

        try:
            self.model = YOLO(weights)
        except Exception as exc:  # noqa: BLE001 -- surface a usable message
            raise RuntimeError(
                f"could not load weights {weights!r}. The first run downloads them, "
                f"so this machine needs internet access; note that yolo11* names "
                f"require ultralytics >= 8.3 (older releases only know yolov8*)."
            ) from exc
        self.model.to(self.device)

        # Read the taxonomy off the checkpoint rather than assuming COCO's
        # numbering, so stock weights and Indian fine-tunes both work here.
        self.taxonomy = build_taxonomy(getattr(self.model, "names", None))

        # Ask YOLO for only the classes we care about. This filtering happens
        # inside the model's postprocessing, so it is cheaper than detecting
        # every class the checkpoint knows and throwing most of them away.
        self._keep_ids = sorted(self.taxonomy)

        self._frame_index = -1
        self._tracking_engaged = False
        self._source_fps = -1.0

    # -- what these particular weights can actually see ---------------------

    @property
    def trained_classes(self) -> set[str]:
        """Names from SENTI_CLASSES the loaded weights can actually produce."""
        return set(self.taxonomy.values())

    @property
    def missing_classes(self) -> set[str]:
        """Names the rule engine knows about that these weights cannot detect.

        On stock COCO weights this is {"auto_rickshaw"}. A rule that depends on
        a name in here will never fire -- check this at startup and fail loudly
        rather than shipping a three-wheeler rule that silently never triggers.
        """
        return set(SENTI_CLASSES) - self.trained_classes

    @staticmethod
    def _auto_device() -> str:
        if torch is None:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _normalise_device(device) -> str:
        """Accept the shapes people actually type and return a canonical string.

        "0" / 0 -> "cuda:0", "GPU" -> "cuda". Without this, `--device cuda:0`
        or `--device 0` quietly disabled FP16, because the old code compared the
        raw string to the literal "cuda".
        """
        text = str(device).strip().lower()
        if text.isdigit():
            return f"cuda:{text}"
        if text == "gpu":
            return "cuda"
        return text

    # -- core inference -----------------------------------------------------

    def detect(self, frame, timestamp_ms: float = -1.0) -> FrameResult:
        """Run detection on a single frame. Stateless -- no memory of the past.

        Use this when each frame is judged on its own. Pass timestamp_ms when
        you have it; run_stream does this for you.
        """
        return self._infer(frame, tracking=False, timestamp_ms=timestamp_ms)

    def track(self, frame, timestamp_ms: float = -1.0) -> FrameResult:
        """Detect AND assign a persistent ID to each object across frames.

        NOTE: the tracking half of this is NOT machine learning. Ultralytics
        runs ByteTrack under the hood -- a Kalman filter predicting where each
        box will be next frame, plus Hungarian-algorithm matching of those
        predictions to fresh detections. Classical math, no weights, no
        learning. It is bundled here purely because ultralytics exposes it on
        the same call.

        Stateful rules (red-light jumping, wrong-way) require these IDs: you
        cannot prove a vehicle crossed a line without knowing it is the same
        vehicle that was behind it a second ago.
        """
        return self._infer(frame, tracking=True, timestamp_ms=timestamp_ms)

    def _infer(self, frame, tracking: bool, timestamp_ms: float = -1.0) -> FrameResult:
        """Shared body of detect() and track().

        The two differ by one ultralytics call and nothing else; keeping them as
        two near-identical copies meant a threshold or kwarg fixed in one path
        stayed broken in the other.
        """
        if tracking:
            self._tracking_engaged = True
        elif self._tracking_engaged:
            # ultralytics registers the tracker as a *model-level* callback, so
            # once track() has run, plain predict() calls still go through
            # ByteTrack. The results are not the stateless ones detect()
            # promises. Say so rather than returning quietly-wrong output.
            warnings.warn(
                "detect() called on a detector that has already tracked: "
                "ultralytics keeps its tracker callbacks registered on the "
                "model, so this frame is still being tracked. Use a separate "
                "TrafficDetector for stateless detection.",
                RuntimeWarning,
                stacklevel=3,
            )

        self._frame_index += 1
        t0 = time.perf_counter()

        kwargs = dict(
            imgsz=self.imgsz,
            device=self.device,
            classes=self._keep_ids,
            conf=CONF_FLOOR,       # coarse floor; per-class thresholds in _parse
            verbose=False,
            **self._precision_kwargs,
        )
        if tracking:
            results = self.model.track(
                frame,
                persist=True,          # keep tracker state between calls
                tracker="bytetrack.yaml",
                **kwargs,
            )
        else:
            results = self.model.predict(frame, **kwargs)

        elapsed = (time.perf_counter() - t0) * 1000.0
        dets = self._parse(results[0], self._frame_index)
        return FrameResult(
            frame_index=self._frame_index,
            detections=dets,
            inference_ms=elapsed,
            timestamp_ms=timestamp_ms,
            source_fps=self._source_fps,
        )

    def reset(self) -> None:
        """Forget the frame counter and all tracker state.

        Call this between clips. Without it the frame index keeps climbing
        across unrelated videos, and ByteTrack tries to match the last frame of
        clip A against the first frame of clip B -- producing track IDs that
        span two different scenes, which is exactly the kind of thing a challan
        should never be built on.
        """
        self._frame_index = -1
        self._source_fps = -1.0
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    # -- postprocessing -----------------------------------------------------

    def _parse(self, result, frame_index: int) -> list[Detection]:
        """Turn raw YOLO tensors into our Detection objects, applying the
        per-class thresholds and the minimum-size filter."""
        out: list[Detection] = []

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        # track_id exists only on .track() calls, and is None for boxes the
        # tracker has not yet confirmed.
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        else:
            ids = [None] * len(confs)

        for box, conf, cls_id, tid in zip(xyxy, confs, clses, ids):
            name = self.taxonomy.get(int(cls_id))
            if name is None:
                continue

            # per-class threshold -- the whole point of the coarse floor above
            if float(conf) < CLASS_CONF_THRESHOLDS.get(name, DEFAULT_CONF):
                continue

            x1, y1, x2, y2 = (float(v) for v in box)
            if (x2 - x1) * (y2 - y1) < self.min_box_area:
                continue

            out.append(
                Detection(
                    cls_name=name,
                    confidence=float(conf),
                    xyxy=(x1, y1, x2, y2),
                    track_id=int(tid) if tid is not None else None,
                    frame_index=frame_index,
                )
            )

        return out

    # -- stream helper ------------------------------------------------------

    def run_stream(self, source, use_tracking: bool = True):
        """Yield (frame, FrameResult) for every frame of a video, RTSP URL, or webcam.

        The frame is handed back alongside the result because callers almost
        always need both -- the evidence buffer stores the pixels, the rule
        engine reads the detections.

        The source string is the ONLY difference between the MVP and a live
        deployment:
            "clip.mp4"                       -> uploaded video  (MVP)
            "rtsp://10.0.0.5:554/stream1"    -> live camera     (production)
            0                                -> webcam
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"could not open video source: {source!r}")

        # Live sources buffer frames inside OpenCV. If inference is slower than
        # the camera, that buffer grows and the pipeline drifts further and
        # further behind real time -- for evidence, late is as bad as wrong.
        # Ask for the shallowest buffer the backend will give us.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Each stream is its own scene: restart the frame counter and drop any
        # tracker state left over from the previous clip.
        self.reset()

        fps = cap.get(cv2.CAP_PROP_FPS)
        # Cameras and some containers report 0, NaN, or absurd values.
        self._source_fps = float(fps) if fps and 0 < fps < 1000 else -1.0

        wall_t0 = time.perf_counter()

        # Choose the clock ONCE, before the first frame.
        #
        # Deciding per frame looked reasonable and was wrong: frame 0 of a file
        # legitimately sits at POS_MSEC 0.0, a falsy value that reads as "no
        # timestamp available" and fell back to wall-clock. The result was a
        # timeline that started at whatever the model took to warm up and then
        # jumped *backwards* to 16.7ms on frame 1. A rule measuring elapsed time
        # across that seam sees negative duration.
        #
        # A positive frame count means a seekable recording, which is exactly
        # the case where the container clock is authoritative. Live feeds report
        # 0 or -1 and get wall-clock, which is what "when did this happen" means
        # for a camera anyway.
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        use_container_clock = bool(frame_count and frame_count > 0)
        seq = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if use_container_clock:
                    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if pos_ms is not None and pos_ms >= 0:
                        timestamp_ms = float(pos_ms)
                    elif self._source_fps > 0:
                        # Some codecs drop POS_MSEC mid-file; derive it from the
                        # frame number rather than switching clocks.
                        timestamp_ms = 1000.0 * seq / self._source_fps
                    else:
                        timestamp_ms = (time.perf_counter() - wall_t0) * 1000.0
                else:
                    timestamp_ms = (time.perf_counter() - wall_t0) * 1000.0
                seq += 1

                result = (
                    self.track(frame, timestamp_ms)
                    if use_tracking
                    else self.detect(frame, timestamp_ms)
                )
                yield frame, result
        finally:
            cap.release()


# ---------------------------------------------------------------------------
# 4. VISUALISATION (debug only -- not part of the pipeline)
# ---------------------------------------------------------------------------

_COLOURS = {
    "person": (0, 200, 255),
    "motorcycle": (0, 0, 255),
    "bicycle": (0, 128, 255),
    "car": (0, 255, 0),
    "bus": (255, 128, 0),
    "truck": (255, 0, 128),
    "auto_rickshaw": (0, 255, 255),
}


def draw(frame, result: FrameResult):
    """Draw boxes on a copy of the frame so you can see what the model sees."""
    canvas = frame.copy()
    for d in result.detections:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        colour = _COLOURS.get(d.cls_name, (200, 200, 200))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

        tag = d.cls_name
        if d.track_id is not None:
            tag += f" #{d.track_id}"
        tag += f" {d.confidence:.2f}"

        cv2.putText(
            canvas, tag, (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"frame {result.frame_index}  |  {len(result.detections)} objects  "
        f"|  {result.inference_ms:.1f} ms",
        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return canvas


# ---------------------------------------------------------------------------
# 5. DEMO
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Senti Traffic -- ML detection layer")
    ap.add_argument("--source", required=True,
                    help="video file path, RTSP URL, or webcam index (0)")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="cuda | cpu | 0 (default: auto)")
    ap.add_argument("--min-box-area", type=float, default=400.0,
                    help="reject boxes smaller than this many px^2")
    ap.add_argument("--half", dest="half", action="store_true", default=None,
                    help="force FP16 (CUDA only; ignored elsewhere)")
    ap.add_argument("--no-half", dest="half", action="store_false",
                    help="force FP32 even on CUDA")
    ap.add_argument("--show", action="store_true", help="display annotated video")
    ap.add_argument("--no-track", action="store_true", help="detection only, no IDs")
    args = ap.parse_args()

    try:
        source = int(args.source)      # webcam index
    except ValueError:
        source = args.source           # file path or RTSP URL

    detector = TrafficDetector(
        weights=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        min_box_area=args.min_box_area,
        half=args.half,
    )
    print(f"[senti] model={args.weights}  device={detector.device}  imgsz={args.imgsz}")
    print(f"[senti] detects: {', '.join(sorted(detector.trained_classes))}")
    if detector.missing_classes:
        # Loud on purpose: a rule written against one of these will never fire.
        print(
            f"[senti] NOT detectable with these weights: "
            f"{', '.join(sorted(detector.missing_classes))}"
            f"  -> fine-tune with senti_traffic_train.py"
        )

    total_ms = 0.0
    frames = 0
    seen_tracks: set[int] = set()

    # closing() matters: quitting with 'q' breaks out mid-generator, and without
    # this the VideoCapture -- an open camera or file handle -- is only released
    # whenever the generator happens to be collected.
    stream = detector.run_stream(source, use_tracking=not args.no_track)
    with contextlib.closing(stream):
        for frame, result in stream:
            frames += 1
            total_ms += result.inference_ms
            seen_tracks.update(
                d.track_id for d in result.detections if d.track_id is not None
            )

            if frames % 30 == 0:
                print(
                    f"  frame {result.frame_index:5d}  "
                    f"objects={len(result.detections):3d}  "
                    f"two-wheelers={len(result.two_wheelers):2d}  "
                    f"persons={len(result.persons):2d}  "
                    f"{result.inference_ms:5.1f} ms"
                )

            if args.show:
                cv2.imshow("senti-traffic :: ML layer", draw(frame, result))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if args.show:
        cv2.destroyAllWindows()

    if not frames:
        print(f"[senti] no frames read from {source!r} -- check the path or URL")
        return

    avg_ms = total_ms / frames
    fps = f"{1000.0 / avg_ms:.1f} FPS" if avg_ms > 0 else "n/a"
    print(
        f"\n[senti] {frames} frames  |  avg {avg_ms:.1f} ms/frame  ({fps})  |  "
        f"{len(seen_tracks)} unique tracks"
    )


if __name__ == "__main__":
    main()
