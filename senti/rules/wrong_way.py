"""
senti.rules.wrong_way
=====================
Reference implementation. Every other rule follows this shape.

Wrong-way needs no extra model, no signal state and no homography -- only a
track's direction of travel compared against the direction its lane is supposed
to flow. Pure geometry.

WHY THIS RULE IS PER-LANE AND NOT PER-CAMERA
The first live test used one `allowed_heading` for the whole frame and produced
EIGHT false positives on a two-way road: vehicles travelling perfectly legally in
the opposite carriageway were flagged, because a single direction cannot describe
a road where traffic moves both ways. No value of that setting fixes it.

Direction is a property of the LANE. So the rule now asks the calibration which
lane a vehicle is in and compares against that lane's heading.

AND WHEN THERE IS NO LANE, IT ABSTAINS
If the vehicle's contact point falls outside every declared lane, `evaluate`
returns None. Not "probably fine", not a guess against some default -- no verdict
at all. For a system that ends in a fine, silence is cheap and a wrongful challan
is not. The same reasoning applies when the camera has no calibration: every
geometric rule abstains rather than inventing geometry.

CALIBRATION (written by scripts/calibrate.py)

    calibration:
      lanes:
        - name: northbound
          polygon: [[420,700],[900,700],[980,1050],[350,1050]]
          heading: [0, -1]          # image coords: y increases DOWNWARD
        - name: southbound
          polygon: [[1000,700],[1480,700],[1540,1050],[1010,1050]]
          heading: [0, 1]
"""

from __future__ import annotations

import math
from typing import Optional

from ..core.types import Detection, FrameResult
from .base import Rule


class WrongWayRule(Rule):
    name = 'wrong_way'
    applies_to = ('car', 'motorcycle', 'auto_rickshaw', 'bus', 'truck',
                  'commercial_vehicle', 'bicycle', 'tractor')
    requires = ('calibration',)
    stateful = True          # needs track history to have a direction at all
    min_frames = 8
    cooldown_frames = 200

    mv_act_section = 'MV Act s.184 (driving dangerously)'
    description = 'Vehicle travelling against the permitted direction of its lane'

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)

        # A vehicle counts as wrong-way only once it is well past perpendicular
        # to the permitted flow. 70 degrees leaves room for turns and lane changes.
        self.tolerance_deg = float(self.config.get('tolerance_deg', 70))

        # Ignore vehicles that have barely moved. A stationary vehicle's heading
        # is box jitter, and jitter points in random directions -- without this
        # every parked vehicle eventually looks like a wrong-way driver.
        self.min_displacement = float(self.config.get('min_displacement', 40))

        # Fallback heading, used ONLY when the camera has no lanes drawn. Off by
        # default: guessing a direction is what produced the original false
        # positives, so it must be opted into explicitly.
        fb = self.config.get('fallback_heading')
        if fb:
            mag = math.hypot(fb[0], fb[1]) or 1.0
            self.fallback = (fb[0] / mag, fb[1] / mag)
        else:
            self.fallback = None

        self._warned = False

    # ---------------------------------------------------------------------

    def evaluate(self, det: Detection, result: FrameResult,
                 context: dict) -> Optional[tuple[str, dict]]:
        tid = det.track_id

        # 1. has it actually moved enough to have a direction?
        disp = self.displacement(tid)
        if disp < self.min_displacement:
            return None

        heading = self.heading(tid)
        if heading is None:
            return None

        # 2. which lane is it in? Test the CONTACT POINT with the road, not the
        #    box centre -- a bus's centre floats metres above the tarmac and
        #    would fall in the wrong lane on any angled view.
        cal = context.get('calibration')
        lane = cal.lane_at(det.bottom_center) if (cal and cal.is_calibrated) else None

        if cal is not None and cal.is_calibrated and lane is None:
            # Outside every declared lane -- footpath, opposite carriageway,
            # median, or simply an area nobody calibrated. ABSTAIN.
            return None

        if lane is not None:
            permitted = lane.heading
            lane_name = lane.name
            angle = lane.angle_from_heading(heading)
        else:
            # No calibration at all for this camera.
            if self.fallback is None:
                if not self._warned:
                    print('[wrong_way] no calibration and no fallback_heading -- '
                          'abstaining. Run scripts/calibrate.py to draw lanes.')
                    self._warned = True
                return None
            permitted = self.fallback
            lane_name = None
            dot = max(-1.0, min(1.0,
                                heading[0] * permitted[0] + heading[1] * permitted[1]))
            angle = math.degrees(math.acos(dot))

        # 3. is it far enough off the permitted direction?
        if angle <= self.tolerance_deg:
            return None

        where = f'in lane "{lane_name}"' if lane_name else 'on an uncalibrated approach'
        reason = (
            f'{det.cls_name} (track #{tid}) travelling {angle:.0f} degrees against '
            f'the permitted direction of flow {where}, sustained over '
            f'{disp:.0f}px of movement'
        )

        # margin_norm feeds the Evidence Defensibility Score: 70 deg is the
        # threshold and 180 is a complete reversal, so scale between them. A
        # vehicle that only just crossed the threshold is weaker evidence than
        # one driving squarely head-on.
        margin = (angle - self.tolerance_deg) / max(1.0, 180.0 - self.tolerance_deg)

        return reason, {
            'angle_deg': round(angle, 1),
            'lane': lane_name,
            'heading': [round(heading[0], 3), round(heading[1], 3)],
            'permitted_heading': [round(permitted[0], 3), round(permitted[1], 3)],
            'displacement_px': round(disp, 1),
            'margin_norm': round(max(0.0, min(1.0, margin)), 3),
            'calibrated': lane_name is not None,
        }
