"""
senti.rules.lane_discipline
===========================
Slow-moving vehicles obstructing the fast lane.

WHY THE THRESHOLD IS RELATIVE, NOT ABSOLUTE
"Slower than 60 km/h" is meaningless on its own: 60 is obstructive on a clear
expressway and perfectly normal in congestion. The rule therefore compares a
vehicle against the MEDIAN SPEED OF OTHER TRAFFIC in the same frame. That makes
it self-calibrating -- during a jam every vehicle is slow, the median drops with
them, and nobody is flagged. Which is correct: in a jam, nobody is obstructing.

Median rather than mean, so one stopped vehicle cannot drag the reference down.

INDIA DRIVES ON THE LEFT
The fast lane is the RIGHTMOST lane. Keeping left except when overtaking is the
rule, so a slow vehicle sitting in the right lane is the violation. Which lane is
"fast" is declared per camera -- never inferred from position, because camera
angle makes rightmost-in-frame unreliable.

⚠️ ADVISORY, NOT A CHALLAN
The legal basis for lane discipline in India is thinner than for helmet or
signal violations -- generally MV Act s.177 (general offence) rather than a
dedicated provision. This rule is built to FLAG for officer attention, and marks
itself `advisory: true` so the portal can present it differently from an
enforceable violation.

CONFIG
    calibration:
      lanes:
        - name: lane3_right
          polygon: [...]
          heading: [0, -1]
          fast_lane: true          # <- declare it

    rules:
      lane_discipline:
        slow_ratio: 0.7            # below 70% of median traffic speed
        min_reference_vehicles: 3  # need enough others to form a median
        min_speed_kmph: 20         # ignore near-stationary (that is congestion)
        min_frames: 25             # ~1s at 25fps; brief overtakes are legitimate
"""

from __future__ import annotations

import statistics
from typing import Optional

from ..core.types import Detection, FrameResult
from .base import Rule


class LaneDisciplineRule(Rule):
    name = 'lane_discipline'
    applies_to = ('car', 'motorcycle', 'auto_rickshaw', 'bus', 'truck',
                  'commercial_vehicle', 'tractor')
    requires = ('calibration.homography', 'calibration.lanes')
    stateful = True
    min_frames = 25               # a brief overtake is not obstruction
    cooldown_frames = 400

    mv_act_section = 'MV Act s.177 (general offence) - ADVISORY'
    description = 'Slow-moving vehicle obstructing the fast lane'

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.slow_ratio = float(self.config.get('slow_ratio', 0.7))
        self.min_reference = int(self.config.get('min_reference_vehicles', 3))
        self.min_speed = float(self.config.get('min_speed_kmph', 20.0))
        self.window = int(self.config.get('window_frames', 12))
        self._warned = False

    def evaluate(self, det: Detection, result: FrameResult,
                 context: dict) -> Optional[tuple[str, dict]]:
        cal = context.get('calibration')
        hom = getattr(cal, 'homography', None) if cal else None

        if cal is None or not cal.is_calibrated or hom is None:
            if not self._warned:
                print('[lane_discipline] needs lanes AND a homography -- abstaining. '
                      'Draw lanes and 4 homography points with scripts/calibrate.py, '
                      'and mark the fast lane with `fast_lane: true`.')
                self._warned = True
            return None

        lane = cal.lane_at(det.bottom_center)
        if lane is None or not getattr(lane, 'fast_lane', False):
            return None                       # not in the fast lane -> not this rule

        own = self.speed_kmph(det.track_id, hom, window=self.window)
        if own is None or own < self.min_speed:
            # near-stationary means congestion, not obstruction
            return None

        # reference: every OTHER tracked vehicle we can measure this frame
        others = []
        for d in result.detections:
            if d.track_id is None or d.track_id == det.track_id:
                continue
            if not d.is_vehicle:
                continue
            s = self.speed_kmph(d.track_id, hom, window=self.window)
            if s is not None and s >= self.min_speed:
                others.append(s)

        if len(others) < self.min_reference:
            return None                       # not enough traffic to judge against

        median = statistics.median(others)
        threshold = median * self.slow_ratio
        if own >= threshold:
            return None

        ratio = own / median if median else 1.0
        reason = (
            f'{det.cls_name} (track #{det.track_id}) travelling {own:.0f} km/h in '
            f'fast lane "{lane.name}" while surrounding traffic averages '
            f'{median:.0f} km/h - {ratio*100:.0f}% of prevailing speed, obstructing '
            f'overtaking traffic. Advisory only.'
        )

        # the further below the threshold, the stronger the case
        margin = min(1.0, (threshold - own) / max(1.0, threshold))

        return reason, {
            'speed_kmph': round(own, 1),
            'median_traffic_kmph': round(median, 1),
            'ratio_of_median': round(ratio, 3),
            'reference_vehicles': len(others),
            'lane': lane.name,
            'margin_norm': round(margin, 3),
            'advisory': True,
            'note': 'Lane-discipline enforcement in India rests on MV Act s.177; '
                    'flag for officer attention rather than automatic challan.',
        }
