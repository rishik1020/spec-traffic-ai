"""
senti.rules.wrong_way
=====================
Reference implementation. Every other rule follows this shape.

Wrong-way is the right rule to build first: it needs no extra model, no signal
state, and no homography -- only a track's direction of travel compared against
the direction that approach is supposed to flow. Pure geometry.

CALIBRATION
One vector per camera, in config:

    rules:
      wrong_way:
        allowed_heading: [0, -1]     # up the frame, in image coordinates
        tolerance_deg: 70
        min_displacement: 40

Note image coordinates: y increases DOWNWARD. [0, -1] means "moving toward the
top of the frame".
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
    requires = ()
    stateful = True          # needs track history to know a direction at all
    min_frames = 8
    cooldown_frames = 200

    mv_act_section = 'MV Act s.184 (driving dangerously)'
    description = 'Vehicle travelling against the permitted direction of flow'

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        heading = self.config.get('allowed_heading', [0, -1])
        mag = math.hypot(heading[0], heading[1]) or 1.0
        self.allowed = (heading[0] / mag, heading[1] / mag)

        # A vehicle is only "wrong way" if it is well past perpendicular to the
        # permitted flow. 70 deg leaves room for turning and lane changes.
        self.tolerance = math.cos(math.radians(
            float(self.config.get('tolerance_deg', 70))))

        # Ignore vehicles that have barely moved. A stationary car's heading is
        # box jitter, and jitter points in random directions -- without this
        # every parked vehicle eventually looks like a wrong-way driver.
        self.min_displacement = float(self.config.get('min_displacement', 40))

    def evaluate(self, det: Detection, result: FrameResult,
                 context: dict) -> Optional[tuple[str, dict]]:
        tid = det.track_id

        if self.displacement(tid) < self.min_displacement:
            return None

        heading = self.heading(tid)
        if heading is None:
            return None

        # dot product of two unit vectors = cos of the angle between them
        dot = heading[0] * self.allowed[0] + heading[1] * self.allowed[1]
        if dot >= self.tolerance:
            return None                      # travelling acceptably

        angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
        reason = (
            f'{det.cls_name} (track #{tid}) travelling {angle:.0f} degrees '
            f'against the permitted direction of flow, sustained over '
            f'{self.displacement(tid):.0f}px of movement'
        )
        return reason, {
            'angle_deg': round(angle, 1),
            'heading': [round(heading[0], 3), round(heading[1], 3)],
            'allowed_heading': [round(self.allowed[0], 3), round(self.allowed[1], 3)],
            'displacement_px': round(self.displacement(tid), 1),
        }
