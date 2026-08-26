"""
senti.rules.over_speeding
=========================
Speed from monocular video, via the per-camera homography.

⚠️ THIS IS A SCREENING SIGNAL, NOT LEGAL EVIDENCE
Speed challans in India require a type-approved radar or LIDAR device. A camera
estimating speed from pixel motion cannot support a fine on its own, and this
rule says so in every evidence package it produces (`legally_admissible: false`).

What it IS good for: flagging vehicles for officer attention, measuring the speed
distribution of a corridor, and correlating speed against other violations.

The defensible camera-only alternative is point-to-point average speed between two
calibrated lines -- distance over elapsed time, which measures duration rather
than instantaneous velocity. That is future work; see STATUS.md.

LIMITS ARE PER VEHICLE CLASS, NOT PER ROAD
The Motor Vehicles Act sets different limits by vehicle category, so a truck at
100 km/h on an expressway is speeding while a car at 100 is not. Encoding one
number per road would be wrong for India, where a single carriageway carries
cars, buses, trucks and (where permitted) two-wheelers under different limits.

CONFIG
    rules:
      over_speeding:
        limits_kmph:                 # per detected class
          car: 120
          bus: 80
          truck: 80
          commercial_vehicle: 80
          motorcycle: 80
          auto_rickshaw: 50
        default_limit_kmph: 80
        tolerance_kmph: 5            # enforcement margin for measurement error
        min_frames: 10
"""

from __future__ import annotations

from typing import Optional

from ..core.types import Detection, FrameResult
from .base import Rule

# Sensible starting points for an Indian access-controlled expressway such as
# the Hyderabad ORR. VERIFY against the current state notification before
# quoting these as legal limits -- they vary by state and are revised.
DEFAULT_LIMITS = {
    'car': 120.0,
    'motorcycle': 80.0,
    'auto_rickshaw': 50.0,
    'bus': 80.0,
    'truck': 80.0,
    'commercial_vehicle': 80.0,
    'tractor': 40.0,
    'bicycle': 25.0,
}


class OverSpeedingRule(Rule):
    name = 'over_speeding'
    applies_to = ('car', 'motorcycle', 'auto_rickshaw', 'bus', 'truck',
                  'commercial_vehicle', 'tractor')
    requires = ('calibration.homography',)
    stateful = True
    min_frames = 10               # speed needs a settled track, not a glimpse
    cooldown_frames = 250

    mv_act_section = 'MV Act s.183 (driving at excessive speed) - SCREENING ONLY'
    description = 'Vehicle exceeding the speed limit for its class'

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.limits = dict(DEFAULT_LIMITS)
        self.limits.update(self.config.get('limits_kmph', {}) or {})
        self.default_limit = float(self.config.get('default_limit_kmph', 80.0))

        # Enforcement margin. Vision speed carries roughly 10-15% error, so
        # firing at exactly the limit would flag compliant drivers. Real
        # enforcement applies a tolerance too.
        self.tolerance = float(self.config.get('tolerance_kmph', 5.0))
        self.window = int(self.config.get('window_frames', 12))
        self._warned = False

    def limit_for(self, cls_name: str, lane) -> float:
        """Lane limit wins if set, else the per-class limit, else the default."""
        if lane is not None and getattr(lane, 'speed_limit_kmph', None):
            return float(lane.speed_limit_kmph)
        return float(self.limits.get(cls_name, self.default_limit))

    def evaluate(self, det: Detection, result: FrameResult,
                 context: dict) -> Optional[tuple[str, dict]]:
        cal = context.get('calibration')
        hom = getattr(cal, 'homography', None) if cal else None

        if hom is None:
            if not self._warned:
                print('[over_speeding] no homography for this camera -- abstaining. '
                      'Run scripts/calibrate.py, press H, and click four road '
                      'points with known real-world spacing.')
                self._warned = True
            return None

        kmph = self.speed_kmph(det.track_id, hom, window=self.window)
        if kmph is None:
            return None                      # not measurable -> no verdict

        lane = cal.lane_at(det.bottom_center) if cal.is_calibrated else None
        limit = self.limit_for(det.cls_name, lane)

        if kmph <= limit + self.tolerance:
            return None

        over = kmph - limit
        where = f' in lane "{lane.name}"' if lane is not None else ''
        reason = (
            f'{det.cls_name} (track #{det.track_id}) measured at {kmph:.0f} km/h '
            f'against a {limit:.0f} km/h limit{where}, exceeding it by {over:.0f} km/h. '
            f'Vision-based estimate - screening signal only, not legal evidence.'
        )

        # margin_norm feeds EDS: 50 km/h over is unambiguous, 5 km/h over is
        # within measurement noise and should score as weak evidence.
        margin = min(1.0, over / 50.0)

        return reason, {
            'speed_kmph': round(kmph, 1),
            'limit_kmph': round(limit, 1),
            'over_by_kmph': round(over, 1),
            'lane': lane.name if lane is not None else None,
            'tolerance_kmph': self.tolerance,
            'margin_norm': round(margin, 3),
            'method': 'monocular homography',
            'legally_admissible': False,
            'note': 'Legal speed enforcement requires radar or LIDAR.',
        }
