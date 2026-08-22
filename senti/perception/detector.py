"""
senti.perception.detector
=========================
The entire ML surface area of SPEC Traffic AI.

Supervision is used HERE and only here -- for detection parsing and filtering.
Everything downstream receives our own `Detection` / `FrameResult` objects.

That boundary is deliberate. supervision's `Detections` is a struct-of-arrays
built for vectorised pipeline work; our rules need per-object domain semantics
(`bottom_center`, `is_two_wheeler`, `riders_on`). Converting once, here, keeps
the rule engine free of any third-party data model -- which is what makes the
claim "swap the detector without touching the rules" actually true.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import supervision as sv
from ultralytics import YOLO

from ..core.types import CLASS_NAMES, UNRELIABLE, Detection, FrameResult

# Per-class confidence floors. Small, frequently-occluded objects in dense
# traffic need a looser threshold or they vanish; large unambiguous ones can
# afford to be strict. Retune from the F1-vs-confidence curve of your run.
CLASS_CONF: dict[str, float] = {
    'pedestrian': 0.30,
    'motorcycle': 0.30,
    'bicycle': 0.30,
    'auto_rickshaw': 0.35,
    'traffic_light': 0.25,      # small and critical -- prefer recall
    'zebra_crossing': 0.30,
    'car': 0.45,
    'bus': 0.45,
    'truck': 0.45,
    'commercial_vehicle': 0.45,
}
DEFAULT_CONF = 0.35

# Floor handed to YOLO itself. Must sit at or below every per-class value and
# below DEFAULT_CONF, or the model silently drops detections before _parse can
# apply the real thresholds.
CONF_FLOOR = min([*CLASS_CONF.values(), DEFAULT_CONF])


class TrafficDetector:
    def __init__(
        self,
        weights: str | Path = 'runs/driveindia/weights/best.pt',
        device: Optional[str] = None,
        imgsz: int = 640,
        min_box_area: float = 400.0,
        half: Optional[bool] = None,
        drop_unreliable: bool = True,
    ) -> None:
        try:
            import torch
            auto = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            auto = 'cpu'

        self.device = device or auto
        self.imgsz = imgsz
        self.min_box_area = min_box_area
        self.half = (self.device == 'cuda') if half is None else half
        self.drop_unreliable = drop_unreliable

        w = Path(weights)
        self.indian_model = w.exists()
        if not self.indian_model:
            print(f'[senti] WARNING: {w} not found -- falling back to COCO yolo11n.pt')
            print('[senti] auto_rickshaw and traffic_light will NOT be detected.')
            w = Path('yolo11n.pt')

        self.model = YOLO(str(w))
        self.model.to(self.device)

        # Trust the checkpoint's own names when present; fall back to our table.
        self.names: dict[int, str] = dict(getattr(self.model, 'names', {})) or CLASS_NAMES
        self._frame_index = -1

    # -- inference ---------------------------------------------------------

    def detect(self, frame, frame_index: int = -1, timestamp: float = 0.0) -> FrameResult:
        return self._run(frame, track=False, frame_index=frame_index, timestamp=timestamp)

    def track(self, frame, frame_index: int = -1, timestamp: float = 0.0) -> FrameResult:
        """Detect and assign persistent ids.

        The tracking half is NOT machine learning -- ByteTrack is a Kalman
        filter predicting each box's next position plus Hungarian matching of
        predictions to fresh detections. Classical maths, no weights.

        Stateful rules need these ids: you cannot prove a vehicle crossed a line
        without knowing it is the same vehicle that was behind it.
        """
        return self._run(frame, track=True, frame_index=frame_index, timestamp=timestamp)

    def _run(self, frame, track: bool, frame_index: int, timestamp: float) -> FrameResult:
        if frame_index < 0:
            self._frame_index += 1
            frame_index = self._frame_index

        t0 = time.perf_counter()
        kwargs = dict(imgsz=self.imgsz, device=self.device,
                      conf=CONF_FLOOR, verbose=False)
        # only pass half when actually enabled -- ultralytics warns on every
        # call otherwise, which floods the log at 25fps
        if self.half:
            kwargs['half'] = True
        if track:
            raw = self.model.track(frame, persist=True, tracker='bytetrack.yaml', **kwargs)
        else:
            raw = self.model.predict(frame, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0

        return FrameResult(
            frame_index=frame_index,
            detections=self._parse(raw[0], frame_index),
            inference_ms=elapsed,
            timestamp=timestamp,
        )

    # -- supervision boundary ---------------------------------------------

    def _parse(self, raw, frame_index: int) -> list[Detection]:
        """YOLO output -> sv.Detections (filtering) -> our Detection objects."""
        dets = sv.Detections.from_ultralytics(raw)
        if len(dets) == 0:
            return []

        # vectorised filters, courtesy of supervision
        keep = np.ones(len(dets), dtype=bool)

        if dets.confidence is not None and dets.class_id is not None:
            thresholds = np.array([
                CLASS_CONF.get(self.names.get(int(c), ''), DEFAULT_CONF)
                for c in dets.class_id
            ])
            keep &= dets.confidence >= thresholds

        areas = dets.box_area if hasattr(dets, 'box_area') else np.array([
            (b[2] - b[0]) * (b[3] - b[1]) for b in dets.xyxy
        ])
        keep &= areas >= self.min_box_area

        dets = dets[keep]
        if len(dets) == 0:
            return []

        out: list[Detection] = []
        tracker_ids = dets.tracker_id if dets.tracker_id is not None else [None] * len(dets)

        for i in range(len(dets)):
            cid = int(dets.class_id[i]) if dets.class_id is not None else -1
            name = self.names.get(cid)
            if name is None:
                continue
            if self.drop_unreliable and name in UNRELIABLE:
                continue

            x1, y1, x2, y2 = (float(v) for v in dets.xyxy[i])
            tid = tracker_ids[i]
            out.append(Detection(
                cls_name=name,
                confidence=float(dets.confidence[i]) if dets.confidence is not None else 0.0,
                xyxy=(x1, y1, x2, y2),
                track_id=int(tid) if tid is not None else None,
                frame_index=frame_index,
            ))
        return out


# ---------------------------------------------------------------------------
# Signal state -- classical CV, deliberately not a model
# ---------------------------------------------------------------------------

def read_signal_state(frame, light: Detection) -> tuple[str, float]:
    """Return (state, score) for one traffic-light box.

    WHY THIS IS NOT MACHINE LEARNING
    The model gives you the signal HEAD's location; it says nothing about which
    lamp is lit. Once you have an isolated ~30x80px crop containing exactly one
    traffic light, HSV thresholding beats a learned classifier: no training
    data, microseconds to run, and -- decisively -- fully explainable when a
    challan is contested. "62% of pixels in the crop matched red" is defensible.
    "The network said red" is not.

    The returned score is that pixel fraction, and it feeds the Evidence
    Defensibility Score directly.
    """
    import cv2

    x1, y1, x2, y2 = (int(v) for v in light.xyxy)
    h, w = frame.shape[:2]
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return 'unknown', 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    masks = {
        # red wraps the hue circle, so it needs two ranges
        'red': cv2.bitwise_or(
            cv2.inRange(hsv, (0, 90, 90), (10, 255, 255)),
            cv2.inRange(hsv, (170, 90, 90), (180, 255, 255))),
        'amber': cv2.inRange(hsv, (11, 90, 90), (32, 255, 255)),
        'green': cv2.inRange(hsv, (40, 60, 60), (90, 255, 255)),
    }

    total = crop.shape[0] * crop.shape[1]
    scores = {k: float(np.count_nonzero(m)) / total for k, m in masks.items()}
    best = max(scores, key=scores.get)

    # a lit lamp occupies a meaningful share of the head; below that it is
    # reflection, brake lights bleeding in, or a dark/failed signal
    if scores[best] < 0.04:
        return 'unknown', scores[best]
    return best, scores[best]
