"""
senti_traffic_v4.py
===================
Senti Traffic -- perception layer, v4 (DriveIndia edition).

Supersedes senti_traffic_ml.py, which was written against COCO's taxonomy
before we had an Indian-trained model. What changed and why:

  v1 (senti_traffic_ml.py)          v4 (this file)
  --------------------------------  --------------------------------------
  COCO 80-class pretrained          DriveIndia fine-tune, 28 classes
  no auto-rickshaw at all           `auto_rickshaw` as a first-class citizen
  no traffic-light class            `traffic_light` -> red-light rule possible
  `person` only from COCO           `pedestrian` in the same model
  needed 2 models for triple-riding ONE model covers every rule

Still the same contract: this file finds things in a frame. It never decides
whether anything is a violation -- that is the rule engine's job. Everything
downstream consumes `Detection` objects and never touches a model.

USAGE
    from senti_traffic_v4 import TrafficDetector

    det = TrafficDetector(weights='results/driveindia/best.pt')
    for frame, result in det.run_stream('junction.mp4'):
        bikes = result.two_wheelers
        riders = result.riders_on(bikes[0]) if bikes else []

CLI
    python senti_traffic_v4.py --source clip.mp4 --show
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# 1. DRIVEINDIA TAXONOMY
# ---------------------------------------------------------------------------
# Ids are inferred from the DriveIndia paper's class ordering cross-checked
# against observed label frequencies -- the release ships no class-name file.
# Verified visually for 0, 3, 9 and 10; the rest are high-confidence.
# Ids 24-27 are undocumented in the paper entirely.

CLASS_NAMES: dict[int, str] = {
    0:  'pedestrian',        1:  'bicycle',              2:  'car',
    3:  'motorcycle',        4:  'route_board',          5:  'bus',
    6:  'commercial_vehicle',7:  'truck',                8:  'traffic_sign',
    9:  'traffic_light',     10: 'auto_rickshaw',        11: 'ambulance',
    12: 'construction_vehicle', 13: 'animal',            14: 'unmarked_speed_bump',
    15: 'marked_speed_bump', 16: 'pothole',              17: 'police_vehicle',
    18: 'tractor',           19: 'pushcart',             20: 'temp_traffic_barrier',
    21: 'rumble_strips',     22: 'traffic_cone',         23: 'zebra_crossing',
    24: 'undocumented_24',   25: 'undocumented_25',      26: 'undocumented_26',
    27: 'undocumented_27',
}

# Classes with zero or near-zero support in the released subset. The model
# emits them but has never meaningfully learned them -- do NOT build a rule
# that depends on any of these until the data situation changes.
UNRELIABLE: frozenset[str] = frozenset({
    'ambulance', 'animal', 'police_vehicle',   # 0 boxes
    'pothole', 'pushcart', 'rumble_strips',    # 1-22 boxes
    'undocumented_26', 'undocumented_27',
})

# Anything a violation can be attributed to.
VEHICLE_CLASSES = frozenset({
    'bicycle', 'car', 'motorcycle', 'bus', 'commercial_vehicle',
    'truck', 'auto_rickshaw', 'tractor', 'construction_vehicle',
})

# Riders are exposed -- helmet and triple-riding rules only look at these.
TWO_WHEELER_CLASSES = frozenset({'motorcycle', 'bicycle'})

# Three-wheelers. Called out separately because Indian enforcement treats them
# as their own vehicle class, and because helmet rules must NOT apply to them.
THREE_WHEELER_CLASSES = frozenset({'auto_rickshaw'})

# Static road furniture -- useful for calibration and geometry, never a subject.
INFRASTRUCTURE_CLASSES = frozenset({
    'route_board', 'traffic_sign', 'traffic_light', 'zebra_crossing',
    'unmarked_speed_bump', 'marked_speed_bump', 'temp_traffic_barrier',
    'traffic_cone', 'rumble_strips', 'pothole',
})

# Per-class confidence floors.
#
# Small, frequently-occluded objects in dense traffic need a looser threshold
# or they vanish; large unambiguous ones can afford to be strict. Retune these
# from the F1-vs-confidence curve of your actual training run rather than
# trusting the defaults.
CLASS_CONF: dict[str, float] = {
    'pedestrian': 0.30,
    'motorcycle': 0.30,
    'bicycle': 0.30,
    'auto_rickshaw': 0.35,
    'traffic_light': 0.25,   # small and critical -- prefer recall
    'zebra_crossing': 0.30,
    'car': 0.45,
    'bus': 0.45,
    'truck': 0.45,
    'commercial_vehicle': 0.45,
}
DEFAULT_CONF = 0.35

# Floor handed to YOLO itself -- must sit at or below every per-class value,
# otherwise the model drops detections before _parse can apply the real
# thresholds. Include DEFAULT_CONF so classes without an entry stay honest.
CONF_FLOOR = min([*CLASS_CONF.values(), DEFAULT_CONF])


# ---------------------------------------------------------------------------
# 2. OUTPUT CONTRACT
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """One detected object in one frame. The ML/rules boundary."""

    cls_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    track_id: Optional[int] = None
    frame_index: int = -1

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Where the object meets the road.

        Rules must test THIS against stop lines and lane polygons, not the box
        centre. A bus's centre floats metres above the tarmac and would cross a
        painted line long before its wheels do.
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

    @property
    def is_three_wheeler(self) -> bool:
        return self.cls_name in THREE_WHEELER_CLASSES

    @property
    def is_reliable(self) -> bool:
        """False for classes the model never had enough data to learn."""
        return self.cls_name not in UNRELIABLE

    def iou(self, other: 'Detection') -> float:
        ax1, ay1, ax2, ay2 = self.xyxy
        bx1, by1, bx2, by2 = other.xyxy
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def overlap_fraction(self, other: 'Detection') -> float:
        """How much of THIS box sits inside `other`. Asymmetric on purpose --
        a rider is small and the motorcycle is large, so IoU is near-useless
        for associating them while this is exactly right."""
        ax1, ay1, ax2, ay2 = self.xyxy
        bx1, by1, bx2, by2 = other.xyxy
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        return (iw * ih) / self.area if self.area > 0 else 0.0


@dataclass
class FrameResult:
    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

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
    def three_wheelers(self) -> list[Detection]:
        return [d for d in self.detections if d.is_three_wheeler]

    @property
    def pedestrians(self) -> list[Detection]:
        return self.of_class('pedestrian')

    @property
    def traffic_lights(self) -> list[Detection]:
        return self.of_class('traffic_light')

    def riders_on(self, vehicle: Detection, min_overlap: float = 0.35) -> list[Detection]:
        """Pedestrian boxes sitting on a vehicle -- the basis of triple riding.

        NOTE this returns candidates, not a verdict. Deciding that three riders
        constitutes a violation under MV Act s.194C is the rule engine's call,
        and it needs to hold across several frames before issuing a challan --
        a single frame will occasionally catch a pedestrian walking past.
        """
        return [p for p in self.pedestrians
                if p.overlap_fraction(vehicle) >= min_overlap]


# ---------------------------------------------------------------------------
# 3. DETECTOR
# ---------------------------------------------------------------------------

class TrafficDetector:
    """The entire ML surface area of Senti Traffic."""

    def __init__(
        self,
        weights: str | Path = 'results/driveindia/best.pt',
        device: Optional[str] = None,
        imgsz: int = 640,
        min_box_area: float = 400.0,
        half: Optional[bool] = None,
        drop_unreliable: bool = True,
    ) -> None:
        """
        weights          your DriveIndia fine-tune. Falls back to COCO
                         yolo11n.pt with a loud warning if missing, so the
                         pipeline is still runnable before training finishes.
        min_box_area     reject boxes below this (px^2). A 12x20 blob far down
                         the road is not usable as evidence even if it is a
                         real vehicle.
        drop_unreliable  discard classes the model never learned (see
                         UNRELIABLE). On by default -- they are pure noise.
        """
        self.device = device or ('cuda' if (torch and torch.cuda.is_available()) else 'cpu')
        self.imgsz = imgsz
        self.min_box_area = min_box_area
        self.half = (self.device == 'cuda') if half is None else half
        self.drop_unreliable = drop_unreliable

        w = Path(weights)
        if not w.exists():
            print(f'[senti] WARNING: {w} not found -- falling back to COCO yolo11n.pt.')
            print('[senti] Indian classes (auto_rickshaw, traffic_light) will NOT be detected.')
            w = Path('yolo11n.pt')
            self._indian = False
        else:
            self._indian = True

        self.model = YOLO(str(w))
        self.model.to(self.device)

        # Trust the checkpoint's own names over our table when present.
        self.names: dict[int, str] = dict(getattr(self.model, 'names', {})) or CLASS_NAMES
        self._frame_index = -1

    # -- inference ----------------------------------------------------------

    def detect(self, frame) -> FrameResult:
        """Single frame, no memory of the past."""
        return self._run(frame, track=False)

    def track(self, frame) -> FrameResult:
        """Detect and assign persistent ids across frames.

        The tracking half is NOT machine learning -- ByteTrack is a Kalman
        filter predicting each box's next position plus Hungarian matching of
        predictions to fresh detections. Classical maths, no weights.

        Stateful rules need these ids: you cannot prove a vehicle crossed a
        line without knowing it is the same vehicle that was behind it.
        """
        return self._run(frame, track=True)

    def _run(self, frame, track: bool) -> FrameResult:
        self._frame_index += 1
        t0 = time.perf_counter()

        kwargs = dict(imgsz=self.imgsz, device=self.device, half=self.half,
                      conf=CONF_FLOOR, verbose=False)
        if track:
            results = self.model.track(frame, persist=True,
                                       tracker='bytetrack.yaml', **kwargs)
        else:
            results = self.model.predict(frame, **kwargs)

        elapsed = (time.perf_counter() - t0) * 1000.0
        return FrameResult(self._frame_index,
                           self._parse(results[0], self._frame_index),
                           elapsed)

    def _parse(self, result, frame_index: int) -> list[Detection]:
        out: list[Detection] = []
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(confs)

        for box, conf, cls_id, tid in zip(xyxy, confs, clses, ids):
            name = self.names.get(int(cls_id))
            if name is None:
                continue
            if self.drop_unreliable and name in UNRELIABLE:
                continue
            if float(conf) < CLASS_CONF.get(name, DEFAULT_CONF):
                continue

            x1, y1, x2, y2 = (float(v) for v in box)
            if (x2 - x1) * (y2 - y1) < self.min_box_area:
                continue

            out.append(Detection(name, float(conf), (x1, y1, x2, y2),
                                 int(tid) if tid is not None else None,
                                 frame_index))
        return out

    # -- stream -------------------------------------------------------------

    def run_stream(self, source, use_tracking: bool = True):
        """Yield (frame, FrameResult) for a video file, RTSP URL, or webcam.

        The source string is the ONLY difference between the MVP and a live
        deployment:
            'clip.mp4'                     -> uploaded video  (MVP)
            'rtsp://10.0.0.5:554/stream1'  -> live camera     (production)
            0                              -> webcam
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f'could not open video source: {source!r}')
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame, (self.track(frame) if use_tracking else self.detect(frame))
        finally:
            cap.release()


# ---------------------------------------------------------------------------
# 4. SIGNAL STATE -- classical CV, deliberately not a model
# ---------------------------------------------------------------------------

def read_signal_state(frame, light: Detection) -> str:
    """Return 'red' | 'amber' | 'green' | 'unknown' for one traffic-light box.

    WHY THIS IS NOT MACHINE LEARNING
    The model gives you the signal HEAD's location; it says nothing about which
    lamp is lit. Once you have an isolated ~30x80px crop containing exactly one
    traffic light, HSV thresholding beats a learned classifier: it needs no
    training data, runs in microseconds, and -- decisively -- it is fully
    explainable when a challan is contested. "62% of pixels in the upper third
    matched red" is defensible. "The network said red" is not.
    """
    x1, y1, x2, y2 = (int(v) for v in light.xyxy)
    h, w = frame.shape[:2]
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return 'unknown'

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # red wraps around the hue circle, so it needs two ranges
    masks = {
        'red':   cv2.bitwise_or(cv2.inRange(hsv, (0, 90, 90), (10, 255, 255)),
                                cv2.inRange(hsv, (170, 90, 90), (180, 255, 255))),
        'amber': cv2.inRange(hsv, (11, 90, 90), (32, 255, 255)),
        'green': cv2.inRange(hsv, (40, 60, 60), (90, 255, 255)),
    }

    total = crop.shape[0] * crop.shape[1]
    scores = {k: float(np.count_nonzero(m)) / total for k, m in masks.items()}
    best = max(scores, key=scores.get)

    # a lit lamp occupies a meaningful share of the head; below that it is
    # reflection, brake lights bleeding in, or a dark/failed signal
    return best if scores[best] >= 0.04 else 'unknown'


# ---------------------------------------------------------------------------
# 5. DEBUG VISUALISATION
# ---------------------------------------------------------------------------

_COLOURS = {
    'pedestrian': (0, 200, 255),   'motorcycle': (0, 0, 255),
    'bicycle': (0, 128, 255),      'car': (0, 255, 0),
    'auto_rickshaw': (255, 0, 255),'bus': (255, 128, 0),
    'truck': (255, 0, 128),        'commercial_vehicle': (200, 100, 0),
    'traffic_light': (0, 255, 255),'zebra_crossing': (180, 180, 180),
}


def draw(frame, result: FrameResult, signal: Optional[str] = None):
    canvas = frame.copy()
    for d in result.detections:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        colour = _COLOURS.get(d.cls_name, (160, 160, 160))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        tag = d.cls_name
        if d.track_id is not None:
            tag += f' #{d.track_id}'
        tag += f' {d.confidence:.2f}'
        cv2.putText(canvas, tag, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    hud = (f'frame {result.frame_index} | {len(result.detections)} objects | '
           f'{result.inference_ms:.1f} ms')
    if signal:
        hud += f' | signal={signal}'
    cv2.putText(canvas, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# 6. DEMO
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description='Senti Traffic v4 -- perception layer')
    ap.add_argument('--source', required=True, help='video path, RTSP URL, or 0')
    ap.add_argument('--weights', default='results/driveindia/best.pt')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default=None)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--no-track', action='store_true')
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    det = TrafficDetector(weights=args.weights, device=args.device, imgsz=args.imgsz)
    print(f'[senti] device={det.device}  indian_model={det._indian}')

    frames = 0
    total_ms = 0.0
    tracks: set[int] = set()
    flagged = 0

    for frame, result in det.run_stream(source, use_tracking=not args.no_track):
        frames += 1
        total_ms += result.inference_ms
        tracks.update(d.track_id for d in result.detections if d.track_id is not None)

        # signal state, if a light is visible
        lights = result.traffic_lights
        signal = read_signal_state(frame, lights[0]) if lights else None

        # a taste of what the rule engine will do -- NOT a violation decision
        for bike in result.two_wheelers:
            if len(result.riders_on(bike)) >= 3:
                flagged += 1

        if frames % 30 == 0:
            print(f'  frame {result.frame_index:5d}  objects={len(result.detections):3d}  '
                  f'2w={len(result.two_wheelers):2d}  auto={len(result.three_wheelers):2d}  '
                  f'ped={len(result.pedestrians):2d}  signal={signal or "-":7s} '
                  f'{result.inference_ms:5.1f} ms')

        if args.show:
            cv2.imshow('senti-traffic v4', draw(frame, result, signal))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if args.show:
        cv2.destroyAllWindows()

    if frames and total_ms > 0:
        print(f'\n[senti] {frames} frames | avg {total_ms/frames:.1f} ms '
              f'({1000.0*frames/total_ms:.1f} FPS) | {len(tracks)} tracks | '
              f'{flagged} triple-rider candidates')


if __name__ == '__main__':
    main()
