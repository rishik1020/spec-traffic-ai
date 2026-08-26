"""
senti.signal.demand
===================
How much traffic is waiting on each arm of a junction.

WHY THIS IS NOT A RULE
Every other module in `senti` answers "did this vehicle break the law?" and
emits a Violation about ONE track. This answers a different question -- "how
much demand is standing on the north arm right now?" -- about a REGION, on a
clock rather than on a track. Forcing it into the Rule interface would mean
inventing a fake subject and a fake offence, so it gets its own package.

WHY PCU AND NOT VEHICLE COUNT
An arm holding forty motorcycles and an arm holding forty buses are not equally
congested; the buses need roughly six times the green. Counting vehicles says
they are identical. Passenger Car Units (IRC factors, already in
`senti.core.types.PCU`) say what they actually cost in road capacity. This is
the most India-specific decision in the module: Western adaptive systems count
vehicles because their traffic is homogeneous enough to get away with it.

WHY "STOPPED" IS MEASURED AS A FRACTION OF THE VEHICLE'S OWN SIZE
A vehicle 200 m from the camera moves a handful of pixels a second even at
60 km/h, so any fixed pixel threshold marks distant traffic as queued and near
traffic as moving. But a vehicle's BOX shrinks with distance by the same
factor. Measuring displacement as a fraction of the box's own height per second
is therefore scale-free: it means "moved less than a sixth of its own length in
a second", which is true or false regardless of distance or camera height.

A homography is still better -- it gives real km/h -- so it is used when
present. The ratio test is the fallback that keeps the feature usable on an
uncalibrated camera.

WHAT THIS DELIBERATELY WILL NOT DO
It will not count vehicles it cannot see. When the queue reaches the far edge
of the drawn zone the real queue is longer than the observed one, and the
reading is flagged `truncated`. A controller that treats a truncated queue as
the whole truth systematically under-serves the worst-congested arm -- exactly
backwards.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..calibration import Calibration, Homography, StopLine
from ..core.types import Detection, FrameResult

Point = tuple[float, float]


@dataclass
class ApproachDemand:
    """What one arm of the junction is carrying, right now."""

    name: str
    queue_pcu: float = 0.0              # PCU standing still -- the congestion
    queue_count: int = 0
    queue_length_m: Optional[float] = None
    present_pcu: float = 0.0            # everything on the arm, moving or not
    present_count: int = 0
    flow_pcu_hr: float = 0.0            # arrivals, extrapolated to an hour
    waiting_s: float = 0.0              # how long this arm has held a queue
    truncated: bool = False             # queue runs off the edge of the zone
    observed_s: float = 0.0             # window the flow figure is based on
    warming_up: bool = True             # too little observed to state a flow

    @property
    def is_empty(self) -> bool:
        return self.present_count == 0

    def to_dict(self) -> dict:
        d = {
            'approach': self.name,
            'queue_pcu': round(self.queue_pcu, 2),
            'queue_count': self.queue_count,
            'present_pcu': round(self.present_pcu, 2),
            'present_count': self.present_count,
            'flow_pcu_hr': round(self.flow_pcu_hr, 1),
            'waiting_s': round(self.waiting_s, 1),
        }
        if self.queue_length_m is not None:
            d['queue_length_m'] = round(self.queue_length_m, 1)
        if self.truncated:
            d['truncated'] = True
        if self.warming_up:
            d['warming_up'] = True
        return d


class DemandMeter:
    """Per-frame demand measurement, one reading per junction arm.

    Fed every frame by the engine; read whenever the controller wants a
    snapshot. Keeps only what it needs: a short position history per track and
    a sliding window of arrivals per arm.
    """

    def __init__(self, calibration: Calibration, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.cal = calibration

        # A vehicle is queued below this speed. 5 km/h is walking pace -- it
        # covers creeping in a jam, which IS congestion, without counting a
        # vehicle that is merely slow.
        self.stopped_kmph = float(cfg.get('stopped_kmph', 5.0))
        # Fallback with no homography: fraction of the vehicle's OWN box height
        # travelled per second. See the module docstring for why this is
        # scale-free where a pixel threshold is not.
        self.stopped_ratio = float(cfg.get('stopped_ratio', 0.15))
        self.stopped_window = int(cfg.get('stopped_window_frames', 10))

        # Arrival flow is averaged over this window. Two minutes is long enough
        # to survive a whole signal cycle -- so the answer does not swing with
        # the phase -- and short enough to track a building peak.
        self.flow_window_s = float(cfg.get('flow_window_s', 120.0))
        # Until this much has been watched there is no flow RATE, only a
        # handful of arrivals. Extrapolating six seconds of traffic to an hour
        # produced a flow ratio of 5.67 on the first test clip -- arithmetically
        # correct and completely meaningless. Below this the flow is reported as
        # zero and the arm is flagged `warming_up`, which stops the controller
        # adapting on noise. One minute is roughly one signal cycle: the
        # shortest window that can contain a representative sample.
        self.min_observation_s = float(cfg.get('min_observation_s', 60.0))
        # A queue whose tail sits this far along the zone is assumed to
        # continue past it.
        self.truncation_ratio = float(cfg.get('truncation_ratio', 0.9))

        self._hist: dict[int, deque] = {}
        self._arrivals: dict[str, deque] = {}
        self._counted: dict[int, str] = {}        # track -> arm already counted
        self._queue_since: dict[str, Optional[float]] = {}
        self._zone_extent: dict[str, float] = {}  # cached, geometry is static
        self._last: dict[str, list[Detection]] = {}
        self._t: float = 0.0
        self._t0: Optional[float] = None

    # -- ingestion ---------------------------------------------------------

    def update(self, result: FrameResult, timestamp: float) -> None:
        self._t = timestamp
        if self._t0 is None:
            self._t0 = timestamp

        by_arm: dict[str, list[Detection]] = {a: [] for a in self.cal.approach_names}

        for det in result.detections:
            if det.track_id is None or not det.is_vehicle or not det.is_reliable:
                continue
            arm = self.cal.approach_at(det.bottom_center)
            if arm is None:                       # outside every declared lane
                continue                          # -> nobody's demand

            by_arm.setdefault(arm, []).append(det)

            bx, by = det.bottom_center
            h = self._hist.setdefault(det.track_id, deque(maxlen=40))
            h.append((bx, by, timestamp, det.height))

            # One arrival per track per arm. A vehicle that later appears on a
            # second arm counts again there, which is correct -- it is new
            # demand on that arm.
            if self._counted.get(det.track_id) != arm:
                self._counted[det.track_id] = arm
                self._arrivals.setdefault(arm, deque()).append((timestamp, det.pcu))

        cutoff = timestamp - self.flow_window_s
        for q in self._arrivals.values():
            while q and q[0][0] < cutoff:
                q.popleft()

        self._last = by_arm

    # -- reading -----------------------------------------------------------

    def read(self) -> dict[str, ApproachDemand]:
        hom = self.cal.homography
        elapsed = self._t - (self._t0 if self._t0 is not None else self._t)
        window = min(self.flow_window_s, max(1e-3, elapsed))

        out: dict[str, ApproachDemand] = {}
        for arm in self.cal.approach_names:
            dets = self._last.get(arm, [])
            stopped = [d for d in dets if self._is_stopped(d, hom)]

            arrived_pcu = sum(p for _, p in self._arrivals.get(arm, ()))
            warming = elapsed < self.min_observation_s

            d = ApproachDemand(
                name=arm,
                queue_pcu=sum(x.pcu for x in stopped),
                queue_count=len(stopped),
                present_pcu=sum(x.pcu for x in dets),
                present_count=len(dets),
                # A flow rate needs a window worth extrapolating from. The
                # QUEUE, by contrast, is an instantaneous count and is valid
                # from the first frame -- which is why the controller can still
                # see congestion while it waits for a flow figure.
                flow_pcu_hr=0.0 if warming else (arrived_pcu / window) * 3600.0,
                observed_s=elapsed,
                warming_up=warming,
            )

            line = self.cal.stop_line_for_approach(arm)
            if stopped and line is not None:
                tail = max(self._dist_to_line(x.bottom_center, line, hom)
                           for x in stopped)
                zone = self._zone_span(arm, line, hom)
                if hom is not None:
                    d.queue_length_m = tail
                if zone > 0:
                    d.truncated = tail >= self.truncation_ratio * zone

            started = self._queue_since.get(arm)
            if d.queue_count > 0:
                if started is None:
                    started = self._t
                    self._queue_since[arm] = started
                d.waiting_s = self._t - started
            else:
                self._queue_since[arm] = None

            out[arm] = d
        return out

    # -- internals ---------------------------------------------------------

    def _is_stopped(self, det: Detection, hom: Optional[Homography]) -> bool:
        """Is this vehicle part of a standing queue?

        Unknown counts as NOT stopped. A vehicle seen for two frames has no
        measurable motion, and guessing "stopped" would inflate an arm's demand
        with every new arrival.
        """
        pts = self._hist.get(det.track_id)
        if pts is None or len(pts) < 3:
            return False

        recent = list(pts)[-self.stopped_window:]
        dt = recent[-1][2] - recent[0][2]
        if dt <= 1e-3:
            return False

        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]

        if hom is not None:
            try:
                metres = hom.distance_m((recent[0][0], recent[0][1]),
                                        (recent[-1][0], recent[-1][1]))
                return (metres / dt) * 3.6 < self.stopped_kmph
            except Exception:
                pass                              # fall through to the ratio test

        # scale-free fallback -- see the module docstring
        box_h = max(1.0, sum(p[3] for p in recent) / len(recent))
        px_per_s = math.hypot(dx, dy) / dt
        return (px_per_s / box_h) < self.stopped_ratio

    def _zone_span(self, arm: str, line: StopLine,
                   hom: Optional[Homography]) -> float:
        """How far back the drawn zone can see, measured from the stop line.

        Geometry is static, so this is computed once per arm. It is the
        yardstick for `truncated`: a queue reaching it has left our frame of
        reference, not necessarily the road.
        """
        if arm in self._zone_extent:
            return self._zone_extent[arm]
        span = 0.0
        for lane in self.cal.lanes_for_approach(arm):
            for pt in lane.polygon:
                span = max(span, self._dist_to_line(pt, line, hom))
        self._zone_extent[arm] = span
        return span

    @staticmethod
    def _dist_to_line(pt: Point, line: StopLine, hom: Optional[Homography]) -> float:
        """Perpendicular distance from a point to the stop line.

        In metres when a homography exists, pixels otherwise. Both units are
        internally consistent -- `truncated` compares one distance against
        another in the same space, so it works either way. Only
        `queue_length_m` is withheld without a homography, because a queue
        length in pixels is not a queue length.
        """
        a, b = line.line
        p = pt
        if hom is not None:
            try:
                a = hom.to_metres(a)
                b = hom.to_metres(b)
                p = hom.to_metres(pt)
            except Exception:
                a, b, p = line.line[0], line.line[1], pt
        vx, vy = b[0] - a[0], b[1] - a[1]
        mag = math.hypot(vx, vy)
        if mag < 1e-9:
            return math.hypot(p[0] - a[0], p[1] - a[1])
        # 2D cross product / |v| -- the same primitive StopLine.side() uses
        return abs(vx * (p[1] - a[1]) - vy * (p[0] - a[0])) / mag
