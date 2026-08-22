"""
senti.evidence.buffer
=====================
The rolling frame buffer -- the core of the evidence layer.

THE PROBLEM
A violation is only *recognised* at the moment it completes. By then the part
that proves it -- the vehicle approaching, the light already red -- is in the
past. Start recording when the rule fires and your clip begins too late to
prove anything.

THE FIX
Continuously hold the last N seconds of frames in memory, overwriting the
oldest. When a rule fires at frame T you already have T-5s in RAM; keep
appending for a few more seconds, then flush the whole window to disk.

    frames in memory:  [ t-5s ................ now ]
                              ^
                       violation fires at t
                              |
            clip written = [t-5s ---- t ---- t+3s]
                            approach  event  departure

WHY IT MATTERS LEGALLY
A single still of a motorcycle past the stop line does not prove it crossed
while the signal was red -- it could have been mid-junction when the phase
changed. The clip proves the sequence. That is the difference between a flag
and a challan that survives being contested.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class BufferedFrame:
    index: int
    timestamp: float
    frame: np.ndarray


class RollingBuffer:
    """A circular buffer of recent frames, plus post-event capture."""

    def __init__(
        self,
        fps: float,
        pre_seconds: float = 5.0,
        post_seconds: float = 3.0,
        downscale: Optional[float] = None,
    ) -> None:
        """
        pre_seconds   how much history to hold. 5s covers a vehicle's approach
                      at urban speeds.
        post_seconds  how long to keep recording after a rule fires.
        downscale     store frames at this scale (e.g. 0.5) to cut memory.
                      A 1080p buffer at 25fps x 5s is ~800 MB at full size;
                      halving the dimensions quarters that. Evidence clips do
                      not need full resolution -- the key frames do, and those
                      are captured separately at full size.
        """
        self.fps = max(1.0, fps)
        self.pre_frames = int(self.fps * pre_seconds)
        self.post_frames = int(self.fps * post_seconds)
        self.downscale = downscale

        self._history: deque[BufferedFrame] = deque(maxlen=self.pre_frames)

        # active captures: key -> (frames_collected, remaining, snapshot)
        self._pending: dict[str, dict] = {}

    # -- ingestion ---------------------------------------------------------

    def _store(self, frame: np.ndarray) -> np.ndarray:
        if self.downscale and self.downscale != 1.0:
            h, w = frame.shape[:2]
            return cv2.resize(frame, (int(w * self.downscale), int(h * self.downscale)))
        return frame

    def push(self, index: int, timestamp: float, frame: np.ndarray) -> None:
        """Feed every frame here, violation or not."""
        bf = BufferedFrame(index, timestamp, self._store(frame))
        self._history.append(bf)

        # any capture already running also wants this frame
        for state in self._pending.values():
            if state['remaining'] > 0:
                state['frames'].append(bf)
                state['remaining'] -= 1

    # -- capture -----------------------------------------------------------

    def start_capture(self, key: str) -> None:
        """Begin an evidence capture: snapshot history, then collect forward.

        Safe to call repeatedly for the same key -- a rule that keeps firing
        while a vehicle is still in frame must not start a second clip.
        """
        if key in self._pending:
            return
        self._pending[key] = {
            'frames': list(self._history),      # the past, already captured
            'remaining': self.post_frames,      # the future, still to come
        }

    def is_capturing(self, key: str) -> bool:
        return key in self._pending

    def ready(self) -> list[str]:
        """Keys whose post-roll has completed and are ready to write."""
        return [k for k, s in self._pending.items() if s['remaining'] <= 0]

    def pop(self, key: str) -> list[BufferedFrame]:
        state = self._pending.pop(key, None)
        return state['frames'] if state else []

    def flush_all(self) -> dict[str, list[BufferedFrame]]:
        """Give up remaining post-roll and return everything.

        Called at end-of-stream so a violation near the last frame still
        produces a (shorter) clip rather than being silently dropped.
        """
        out = {k: s['frames'] for k, s in self._pending.items()}
        self._pending.clear()
        return out

    # -- output ------------------------------------------------------------

    def write_clip(self, frames: list[BufferedFrame], path: Path) -> Optional[Path]:
        if not frames:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].frame.shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h)
        )
        try:
            for bf in frames:
                writer.write(bf.frame)
        finally:
            writer.release()
        return path

    @property
    def memory_mb(self) -> float:
        total = sum(bf.frame.nbytes for bf in self._history)
        total += sum(bf.frame.nbytes
                     for s in self._pending.values() for bf in s['frames'])
        return total / 1e6
