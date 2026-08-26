"""
senti.rules.base
================
The Rule interface, the registry, and temporal consistency.

WHY TEMPORAL CONSISTENCY LIVES HERE AND NOT IN EACH RULE
It is the single biggest false-positive killer in the whole system:

  * one frame will catch a pedestrian walking behind a bike -> "triple riding"
  * a tracker ID switch will attribute a red-light jump to the wrong vehicle
  * a box flickering across a stop line for two frames is jitter, not a crossing

Requiring a condition to hold for N consecutive frames ON THE SAME TRACK turns a
noisy detector into something an officer will actually approve. Putting it in
the base class means every rule inherits it instead of each reimplementing it
badly -- or forgetting.

ADDING A VIOLATION
Subclass Rule, set the class attributes, implement evaluate(), and drop the file
in this package. The registry picks it up automatically; enabling it on a camera
is one line of YAML. Nothing else in the codebase changes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.types import Detection, FrameResult

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RULE_REGISTRY: dict[str, type['Rule']] = {}


def get_rule(name: str) -> type['Rule']:
    if name not in RULE_REGISTRY:
        raise KeyError(f'unknown rule {name!r}. available: {sorted(RULE_REGISTRY)}')
    return RULE_REGISTRY[name]


def available_rules() -> list[str]:
    return sorted(RULE_REGISTRY)


# ---------------------------------------------------------------------------
# Violation
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    """A confirmed violation, ready for evidence packaging."""

    rule_name: str
    track_id: int
    cls_name: str
    frame_index: int
    timestamp: float
    confidence: float
    reason: str                                  # plain language, for the officer
    xyxy: tuple[float, float, float, float]
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable id -- also the evidence folder name."""
        return f'{self.rule_name}_t{self.track_id}_f{self.frame_index}'


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------

class Rule:
    """Base class for every violation.

    Subclasses declare what they inspect and what they need, then implement
    evaluate() as a PURE PER-FRAME PREDICATE: "is this track violating right
    now?" The base class handles counting consecutive frames, firing once, and
    resetting when the condition lapses.
    """

    name: str = 'rule'
    applies_to: tuple[str, ...] = ('*',)        # class names, or '*' for all
    requires: tuple[str, ...] = ()              # extra perception, e.g. 'signal_state'
    stateful: bool = False                      # needs track history?
    min_frames: int = 5                         # temporal consistency threshold
    cooldown_frames: int = 150                  # don't re-fire on the same track

    # legal metadata -- surfaced on the challan
    mv_act_section: str = ''
    description: str = ''

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, 'name', None) and cls.name != 'rule':
            RULE_REGISTRY[cls.name] = cls

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.min_frames = int(self.config.get('min_frames', self.min_frames))
        self.cooldown_frames = int(self.config.get('cooldown_frames', self.cooldown_frames))

        self._streak: dict[int, int] = defaultdict(int)
        self._last_fired: dict[int, int] = {}
        self._history: dict[int, list[tuple[float, float, float]]] = defaultdict(list)

    # -- to implement ------------------------------------------------------

    def evaluate(self, det: Detection, result: FrameResult,
                 context: dict) -> Optional[tuple[str, dict]]:
        """Is `det` violating in THIS frame?

        Return (reason, detail) if yes, None if no. Do not worry about
        consecutive frames, cooldowns or firing -- process() handles all of it.
        """
        raise NotImplementedError

    # -- driven by the engine ---------------------------------------------

    def _subjects(self, result: FrameResult) -> list[Detection]:
        if '*' in self.applies_to:
            cands = result.detections
        else:
            wanted = set(self.applies_to)
            cands = [d for d in result.detections if d.cls_name in wanted]
        # an untracked object cannot be held to temporal consistency, and
        # cannot be attributed to a vehicle on a challan
        return [d for d in cands if d.track_id is not None and d.is_reliable]

    def process(self, result: FrameResult, context: dict) -> list[Violation]:
        violations: list[Violation] = []
        seen: set[int] = set()

        for det in self._subjects(result):
            tid = det.track_id
            seen.add(tid)

            if self.stateful:
                # (x, y, t) -- the timestamp is what makes speed computable.
                # bottom_center is the tyre contact point, the only part of the
                # box that lies on the road plane and so the only valid input
                # to a homography.
                bx, by = det.bottom_center
                self._history[tid].append((bx, by, result.timestamp))
                if len(self._history[tid]) > 120:
                    self._history[tid].pop(0)

            verdict = self.evaluate(det, result, context)

            if verdict is None:
                self._streak[tid] = 0
                continue

            self._streak[tid] += 1
            if self._streak[tid] < self.min_frames:
                continue

            last = self._last_fired.get(tid)
            if last is not None and result.frame_index - last < self.cooldown_frames:
                continue

            reason, detail = verdict
            detail = dict(detail)
            detail['sustained_frames'] = self._streak[tid]
            self._last_fired[tid] = result.frame_index

            violations.append(Violation(
                rule_name=self.name,
                track_id=tid,
                cls_name=det.cls_name,
                frame_index=result.frame_index,
                timestamp=result.timestamp,
                confidence=det.confidence,
                reason=reason,
                xyxy=det.xyxy,
                detail=detail,
            ))

        # tracks that left the frame reset, so a vehicle returning later
        # starts a fresh streak rather than resuming a stale one
        for tid in list(self._streak):
            if tid not in seen:
                self._streak[tid] = 0

        return violations

    # -- helpers for subclasses -------------------------------------------

    def track_history(self, track_id: int) -> list[tuple[float, float]]:
        return self._history.get(track_id, [])

    def heading(self, track_id: int, lookback: int = 15) -> Optional[tuple[float, float]]:
        """Unit direction vector of a track over the last `lookback` points.

        Averaged over a window rather than frame-to-frame: single-frame deltas
        are dominated by box jitter, which would make a stationary vehicle
        appear to move in random directions.
        """
        pts = self._history.get(track_id, [])
        if len(pts) < 2:
            return None
        recent = pts[-lookback:]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        mag = (dx * dx + dy * dy) ** 0.5
        if mag < 1e-6:
            return None
        return (dx / mag, dy / mag)

    def speed_kmph(self, track_id: int, homography, window: int = 12,
                   min_points: int = 6) -> Optional[float]:
        """Ground speed in km/h, or None if it cannot be measured honestly.

        Uses the homography to convert pixel motion into metres on the road
        plane, over a window rather than frame-to-frame: single-frame deltas are
        dominated by box jitter, which at 25fps turns a few pixels of noise into
        tens of km/h.

        Returns None -- never a guess -- when there is no homography, too little
        history, no elapsed time, or an implausible result. A speed the system
        cannot stand behind is worse than no speed at all.

        NOTE: this is a SCREENING signal. Legal speed enforcement in India
        requires radar or LIDAR. Point-to-point average speed between two
        calibrated lines is camera-only and defensible; this is not.
        """
        if homography is None:
            return None
        pts = self._history.get(track_id, [])
        if len(pts) < min_points:
            return None

        recent = pts[-window:]
        dt = recent[-1][2] - recent[0][2]
        if dt <= 1e-3:
            return None

        try:
            metres = homography.distance_m((recent[0][0], recent[0][1]),
                                           (recent[-1][0], recent[-1][1]))
        except Exception:
            return None

        kmph = (metres / dt) * 3.6
        # reject nonsense from an ID switch or a bad calibration rather than
        # reporting 400 km/h with a straight face
        if not (0.0 <= kmph <= 250.0):
            return None
        return kmph

    def displacement(self, track_id: int, lookback: int = 15) -> float:
        pts = self._history.get(track_id, [])
        if len(pts) < 2:
            return 0.0
        recent = pts[-lookback:]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        return (dx * dx + dy * dy) ** 0.5
