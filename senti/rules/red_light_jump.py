"""
senti.rules.red_light_jump
==========================
Crossing the stop line and continuing through the junction while the signal is
red.

WHY A CROSSING IS AWKWARD FOR THE RULE INTERFACE
Every other rule tests a condition that HOLDS for a while -- travelling the
wrong way, carrying three riders. A crossing is an INSTANT: one frame the
vehicle is behind the line, the next it is past. A single-frame verdict is
exactly what the base class's temporal consistency exists to reject, because a
box flickering across a painted line for two frames is jitter, not a crossing.

So the rule latches. The instant is recorded once, and the verdict it produces
is the sustained condition that follows it: "this track crossed on red, is
still on the far side, and is still going". That holds for many frames, so the
base class's N-consecutive-frames requirement does its normal job, and a vehicle
whose box merely wobbled over the line never accumulates the streak.

WHY A JUMP AND AN ENCROACHMENT ARE DIFFERENT OFFENCES
A vehicle that rolls a metre past the line and stops has committed a stop-line
violation -- see `stop_line_crossing`. A vehicle that crosses and keeps going
has run the red light. The two differ only in what happens AFTER the crossing,
which is why this rule requires continued progress before it will fire, and why
the other one requires the opposite.

WHY THE SIGNAL STATE IS COLOUR THRESHOLDING, NOT A CLASSIFIER
See `read_signal_state` in the perception layer. "62% of pixels in the signal
head matched red" survives a contested hearing. "The network said red" does not.
The pixel fraction at the moment of crossing is recorded in the evidence and
feeds the Evidence Defensibility Score as context certainty.

WHAT IT REFUSES TO DO
No calibration, no stop line, or an unreadable signal head -- it abstains. The
amber phase is NOT treated as red: entering on amber is lawful in most
circumstances and deciding otherwise is a policy call, not a vision one.

CALIBRATION
    calibration:
      lanes:
        - name: approach1
          polygon: [...]
          heading: [0, -1]
          approach: north
      stop_lines:
        - name: stop_north
          line: [[180, 640], [880, 640]]
          lanes: [approach1]
"""

from __future__ import annotations

import math
from typing import Optional

from ..core.types import Detection, FrameResult
from .base import Rule


class RedLightJumpRule(Rule):
    name = 'red_light_jump'
    applies_to = ('car', 'motorcycle', 'auto_rickshaw', 'bus', 'truck',
                  'commercial_vehicle', 'bicycle', 'tractor')
    requires = ('calibration', 'signal_state')
    stateful = True               # the previous position is the whole rule
    min_frames = 4
    cooldown_frames = 250

    mv_act_section = 'MV Act s.119 r/w s.177 (disobeying a traffic signal)'
    description = 'Vehicle crossed the stop line and proceeded while the signal was red'

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)

        # How far past the line the vehicle must travel before this counts as
        # going THROUGH rather than stopping over the line. Pixels, because the
        # rule must work without a homography.
        self.min_progress_px = float(self.config.get('min_progress_px', 25))

        # How long after the crossing the verdict stays live. Long enough for
        # the streak to build, short enough that a vehicle stopped beyond the
        # line in the next cycle is not still being judged on the last one.
        self.sustain_frames = int(self.config.get('sustain_frames', 60))

        # A signal read from too few red pixels is a guess. Below this the rule
        # abstains rather than issuing a challan on a maybe.
        self.min_signal_score = float(self.config.get('min_signal_score', 0.06))

        self._events: dict[int, dict] = {}
        self._warned = False

    # ---------------------------------------------------------------------

    def _stop_line(self, cal, det: Detection):
        """The stop line this vehicle is approaching, and its lane."""
        lane = cal.lane_at(det.bottom_center)
        if lane is not None and lane.approach:
            line = cal.stop_line_for_approach(lane.approach)
            if line is not None:
                return line, lane
        lines = cal.stop_lines_for(lane.name if lane else None)
        return (lines[0] if lines else None), lane

    def evaluate(self, det: Detection, result: FrameResult,
                 context: dict) -> Optional[tuple[str, dict]]:
        tid = det.track_id
        cal = context.get('calibration')

        if cal is None or not cal.stop_lines:
            if not self._warned:
                print('[red_light_jump] no stop line calibrated -- abstaining. '
                      'Run scripts/calibrate.py and press S to draw one.')
                self._warned = True
            return None

        line, lane = self._stop_line(cal, det)
        if line is None:
            return None

        hist = self.track_history(tid)
        if len(hist) < 2:
            return None
        prev = (hist[-2][0], hist[-2][1])
        curr = (hist[-1][0], hist[-1][1])

        state = context.get('signal_state', 'unknown')
        score = float(context.get('signal_score', 0.0))
        ev = self._events.get(tid)

        # -- the instant ---------------------------------------------------
        if ev is None and line.crossed(prev, curr):
            if state != 'red' or score < self.min_signal_score:
                # Crossed lawfully, or on a signal we cannot read. Latch a
                # refusal so a later wobble over the line cannot re-arm it.
                self._events[tid] = {'blocked': True}
                return None
            self._events[tid] = {
                'frame': result.frame_index,
                'timestamp': result.timestamp,
                'line': line.name,
                'signal_score': score,
                'sign': 1.0 if line.side(curr) > 0 else -1.0,
                'at': curr,
            }
            ev = self._events[tid]

        if not ev or ev.get('blocked'):
            return None

        # -- the sustained condition that follows it ------------------------
        if result.frame_index - ev['frame'] > self.sustain_frames:
            return None

        side_now = 1.0 if line.side(curr) > 0 else -1.0
        if side_now != ev['sign']:
            # Reversed back behind the line. Not a jump.
            return None

        progress = math.hypot(curr[0] - ev['at'][0], curr[1] - ev['at'][1])
        if progress < self.min_progress_px:
            # Still sitting on the line. That is an encroachment, and
            # `stop_line_crossing` is the rule that judges it.
            return None

        reason = (
            f'{det.cls_name} (track #{tid}) crossed stop line "{ev["line"]}" '
            f'while the signal was RED and continued {progress:.0f}px into the '
            f'junction'
        )

        # margin_norm feeds the Evidence Defensibility Score. Two things make
        # this evidence strong: how unambiguously the signal read red, and how
        # far past the line the vehicle actually went.
        sig_term = min(1.0, ev['signal_score'] / 0.25)
        go_term = min(1.0, progress / (4.0 * self.min_progress_px))
        margin = 0.6 * sig_term + 0.4 * go_term

        if len(self._events) > 400:
            self._events.clear()

        return reason, {
            'stop_line': ev['line'],
            'lane': lane.name if lane else None,
            'approach': lane.approach if lane else None,
            'signal_state_at_crossing': 'red',
            'signal_score_at_crossing': round(ev['signal_score'], 3),
            'crossed_at_frame': ev['frame'],
            'crossed_at_time_s': round(ev['timestamp'], 2),
            'progress_px': round(progress, 1),
            'margin_norm': round(max(0.0, min(1.0, margin)), 3),
        }
