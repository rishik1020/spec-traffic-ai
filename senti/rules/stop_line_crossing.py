"""
senti.rules.stop_line_crossing
==============================
Stopping BEYOND the stop line while the signal is red.

THE OFFENCE THIS IS, AND THE ONE IT IS NOT
A vehicle that crosses the line and keeps going has run the red light --
`red_light_jump` judges that. A vehicle that rolls past the line and comes to
rest there has committed a different and much more common offence: it is sitting
in the pedestrian crossing. The two are separated by exactly one thing, what the
vehicle does AFTER the line, so this rule requires the vehicle to have stopped
and the other requires it to have carried on. They cannot both fire on the same
behaviour.

WHY IT WAITS BEFORE JUDGING
Traffic does not halt the instant a signal turns red; the vehicle at the head of
the queue is often already committed. Flagging it the moment the lamp changes
would punish physics. The rule therefore requires the vehicle to be *settled*
past the line -- stationary, for a sustained period -- before it will say
anything.

WHY "BEYOND" IS DERIVED, NOT CONFIGURED
"Past the stop line" is meaningless until you know which side traffic approaches
from. Rather than asking the operator for yet another hand-entered value, the
side is derived from the lane's permitted heading, which calibration already
records. One less thing to get wrong at 2am on a footbridge.

WHAT IT REFUSES TO DO
Without calibration, a stop line, or a readable signal it abstains. It also
abstains for any vehicle whose contact point falls outside every declared lane,
because without a lane there is no heading, and without a heading "beyond" has
no meaning.

LEGAL BASIS
Section 177 is the general-offence provision; stop-line discipline sits under
state traffic regulations made with it. This is flagged for officer attention
rather than treated as an automatic challan.
"""

from __future__ import annotations

from typing import Optional

from ..core.types import Detection, FrameResult
from .base import Rule


class StopLineCrossingRule(Rule):
    name = 'stop_line_crossing'
    applies_to = ('car', 'motorcycle', 'auto_rickshaw', 'bus', 'truck',
                  'commercial_vehicle', 'bicycle', 'tractor')
    requires = ('calibration', 'signal_state')
    stateful = True
    min_frames = 25               # ~1s settled, not a vehicle still rolling
    cooldown_frames = 300

    mv_act_section = 'MV Act s.177 (general offence) - stop-line discipline'
    description = 'Vehicle stopped beyond the stop line while the signal was red'

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)

        # How far past the line before it counts. A tyre touching the paint is
        # not the offence; a vehicle sitting in the crossing is.
        self.min_beyond_px = float(self.config.get('min_beyond_px', 12))

        # Movement over the recent window, below which the vehicle is "settled".
        # Not zero: a stationary box still jitters by a few pixels a frame.
        self.max_creep_px = float(self.config.get('max_creep_px', 14))
        self.creep_lookback = int(self.config.get('creep_lookback', 12))

        self.min_signal_score = float(self.config.get('min_signal_score', 0.06))
        self._warned = False

    # ---------------------------------------------------------------------

    def evaluate(self, det: Detection, result: FrameResult,
                 context: dict) -> Optional[tuple[str, dict]]:
        tid = det.track_id
        cal = context.get('calibration')

        if cal is None or not cal.stop_lines:
            if not self._warned:
                print('[stop_line_crossing] no stop line calibrated -- '
                      'abstaining. Run scripts/calibrate.py and press S.')
                self._warned = True
            return None

        state = context.get('signal_state', 'unknown')
        score = float(context.get('signal_score', 0.0))
        if state != 'red' or score < self.min_signal_score:
            return None

        # A lane is required: without one there is no heading, and without a
        # heading "beyond the line" has no meaning.
        lane = cal.lane_at(det.bottom_center)
        if lane is None:
            return None

        line = (cal.stop_line_for_approach(lane.approach) if lane.approach
                else None)
        if line is None:
            lines = cal.stop_lines_for(lane.name)
            line = lines[0] if lines else None
        if line is None:
            return None

        contact = det.bottom_center
        beyond = line.beyond_sign(lane.heading)
        side = line.side(contact)

        if side * beyond <= 0:          # still behind the line, correctly
            return None

        distance = line.distance(contact)
        if distance < self.min_beyond_px:
            return None

        creep = self.displacement(tid, lookback=self.creep_lookback)
        if creep > self.max_creep_px:
            # Still moving. If it keeps going it is a red-light jump, and
            # `red_light_jump` owns that verdict.
            return None

        reason = (
            f'{det.cls_name} (track #{tid}) stopped {distance:.0f}px beyond '
            f'stop line "{line.name}" in lane "{lane.name}" while the signal '
            f'was RED'
        )

        # margin_norm feeds the Evidence Defensibility Score: how far past the
        # line, how clearly the signal read red, and how firmly the vehicle had
        # actually stopped.
        dist_term = min(1.0, distance / (5.0 * self.min_beyond_px))
        sig_term = min(1.0, score / 0.25)
        still_term = 1.0 - min(1.0, creep / max(1.0, self.max_creep_px))
        margin = 0.45 * dist_term + 0.35 * sig_term + 0.20 * still_term

        return reason, {
            'stop_line': line.name,
            'lane': lane.name,
            'approach': lane.approach,
            'beyond_px': round(distance, 1),
            'signal_state': 'red',
            'signal_score': round(score, 3),
            'creep_px': round(creep, 1),
            'margin_norm': round(max(0.0, min(1.0, margin)), 3),
        }
