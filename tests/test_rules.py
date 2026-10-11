"""
tests/test_rules.py
===================
Rule logic, verified without footage.

WHY THESE EXIST
Three of the six rules cannot be demonstrated on the video we have. There is no
triple riding in the test clip, and no signalised junction at all, so running
the pipeline proves only that the code does not crash -- which is not the same
as proving it is correct. Worse, a rule that silently never fires looks exactly
like a rule with nothing to find.

So the conditions are constructed directly. Each test builds the geometry a rule
is supposed to react to, feeds it frame by frame, and asserts both halves of the
contract: that it FIRES on the offence, and that it ABSTAINS on the lawful case
that most resembles it. The second half is the one that matters -- a rule that
only ever says yes is useless for a system that issues fines.

RUN
    python tests/test_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from senti.calibration import Calibration           # noqa: E402
from senti.core.types import Detection, FrameResult  # noqa: E402
from senti.rules import get_rule                    # noqa: E402

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(name)
    mark = 'ok  ' if condition else 'FAIL'
    print(f'  [{mark}] {name}' + (f'  -- {detail}' if detail and not condition else ''))


def bike(x: float, y: float, w: float = 60, h: float = 90, tid: int = 1) -> Detection:
    return Detection(cls_name='motorcycle', confidence=0.9,
                     xyxy=(x, y, x + w, y + h), track_id=tid)


def person(x: float, y: float, w: float = 22, h: float = 55,
           conf: float = 0.5, tid: int = 100) -> Detection:
    return Detection(cls_name='pedestrian', confidence=conf,
                     xyxy=(x, y, x + w, y + h), track_id=tid)


def car(x: float, y: float, tid: int, w: float = 70, h: float = 60) -> Detection:
    return Detection(cls_name='car', confidence=0.9,
                     xyxy=(x, y, x + w, y + h), track_id=tid)


def run(rule, frames, context):
    """Feed frames in order; return every violation produced."""
    out = []
    for i, dets in enumerate(frames):
        fr = FrameResult(frame_index=i, detections=dets, timestamp=i / 25.0)
        out.extend(rule.process(fr, context))
    return out


# ---------------------------------------------------------------------------
# triple_riding
# ---------------------------------------------------------------------------

def test_triple_riding():
    print('\ntriple_riding')
    Rule = get_rule('triple_riding')

    # Three people seated on one motorcycle. The bike box is y 200-290, so a
    # rider spanning y 170-225 sits mostly above it with roughly 45% of their
    # box inside -- past the rule's 35% threshold, and with their feet well
    # above the bike's contact point at y=290.
    bk = bike(100, 200)
    riders = [person(104, 170, tid=101), person(118, 168, tid=102),
              person(132, 172, tid=103)]
    offence = [[bk] + riders] * 20

    v = run(Rule({'min_frames': 8}), offence, {})
    check('fires on three riders', len(v) == 1, f'got {len(v)}')
    if v:
        d = v[0].detail
        check('counts all three', d['rider_count'] == 3, str(d.get('rider_count')))
        check('records a margin for the EDS',
              0.0 <= d['margin_norm'] <= 1.0, str(d.get('margin_norm')))

    # Two riders is lawful -- a driver and one pillion.
    lawful = [[bk, riders[0], riders[1]]] * 20
    check('abstains on two riders', len(run(Rule({'min_frames': 8}), lawful, {})) == 0)

    # A pedestrian standing on the road BESIDE the bike. Their box overlaps it
    # by more than the 35% threshold, so overlap alone would count them as a
    # third rider -- but their feet are at y=330, below the bike's contact
    # point at y=290, and nobody rides a motorcycle from underneath it.
    beside = person(104, 260, h=70, tid=104)
    decoy = [[bk, riders[0], riders[1], beside]] * 20
    check('rejects a bystander standing beside the bike',
          len(run(Rule({'min_frames': 8}), decoy, {})) == 0)

    # Three riders for only a few frames is jitter, not an offence.
    brief = ([[bk] + riders] * 3) + ([[bk, riders[0]]] * 15)
    check('abstains when the condition does not hold long enough',
          len(run(Rule({'min_frames': 8}), brief, {})) == 0)


# ---------------------------------------------------------------------------
# shared junction calibration
# ---------------------------------------------------------------------------

def junction() -> Calibration:
    """One northbound approach: traffic travels UP the frame toward y=0.

    The stop line sits at y=300, so "beyond" is y < 300.
    """
    return Calibration.from_dict({
        'lanes': [{
            'name': 'approach',
            'polygon': [[0, 100], [500, 100], [500, 700], [0, 700]],
            'heading': [0, -1],
            'approach': 'north',
        }],
        'stop_lines': [{
            'name': 'stop_north',
            'line': [[0, 300], [500, 300]],
            'lanes': ['approach'],
        }],
    })


RED = {'signal_state': 'red', 'signal_score': 0.30}
GREEN = {'signal_state': 'green', 'signal_score': 0.30}


# ---------------------------------------------------------------------------
# red_light_jump
# ---------------------------------------------------------------------------

def test_red_light_jump():
    print('\nred_light_jump')
    Rule = get_rule('red_light_jump')
    cal = junction()

    def approach_then_through(step=14, n=26, start=360):
        """Drive north, crossing y=300 and continuing well past it."""
        return [[car(200, start - i * step, tid=7)] for i in range(n)]

    ctx_red = {'calibration': cal, **RED}
    v = run(Rule({'min_frames': 3}), approach_then_through(), ctx_red)
    check('fires when crossing on red and continuing', len(v) == 1, f'got {len(v)}')
    if v:
        d = v[0].detail
        check('names the stop line', d['stop_line'] == 'stop_north')
        check('records the signal score at the crossing',
              d['signal_score_at_crossing'] == 0.3)
        check('records progress past the line', d['progress_px'] > 25)

    ctx_green = {'calibration': cal, **GREEN}
    check('abstains on green',
          len(run(Rule({'min_frames': 3}), approach_then_through(), ctx_green)) == 0)

    # Crossing the line and STOPPING is an encroachment, not a jump.
    rolls_and_stops = ([[car(200, 360 - i * 14, tid=8)] for i in range(5)]
                       + [[car(200, 295, tid=8)]] * 20)
    check('abstains when the vehicle stops on the line',
          len(run(Rule({'min_frames': 3}), rolls_and_stops, ctx_red)) == 0)

    # A signal read from almost no red pixels is a guess, not evidence.
    faint = {'calibration': cal, 'signal_state': 'red', 'signal_score': 0.01}
    check('abstains when the signal read is too weak to trust',
          len(run(Rule({'min_frames': 3}), approach_then_through(), faint)) == 0)


# ---------------------------------------------------------------------------
# stop_line_crossing
# ---------------------------------------------------------------------------

def test_stop_line_crossing():
    print('\nstop_line_crossing')
    Rule = get_rule('stop_line_crossing')
    cal = junction()
    ctx_red = {'calibration': cal, **RED}

    # Settled 40px past the line (contact point y=260 against a line at 300).
    settled = [[car(200, 200, tid=9)]] * 40
    v = run(Rule({'min_frames': 20}), settled, ctx_red)
    check('fires on a vehicle settled past the line', len(v) == 1, f'got {len(v)}')
    if v:
        d = v[0].detail
        check('measures how far past', d['beyond_px'] > 12, str(d.get('beyond_px')))
        check('names the lane', d['lane'] == 'approach')

    # Correctly stopped BEHIND the line.
    behind = [[car(200, 330, tid=10)]] * 40
    check('abstains behind the line',
          len(run(Rule({'min_frames': 20}), behind, ctx_red)) == 0)

    # Past the line but still moving -- that is a jump, judged elsewhere.
    moving = [[car(200, 240 - i * 10, tid=11)] for i in range(40)]
    check('abstains while the vehicle is still moving',
          len(run(Rule({'min_frames': 20}), moving, ctx_red)) == 0)

    # Same position, green signal: simply waiting in the junction.
    ctx_green = {'calibration': cal, **GREEN}
    check('abstains on green',
          len(run(Rule({'min_frames': 20}), settled, ctx_green)) == 0)


# ---------------------------------------------------------------------------
# the abstention contract every geometric rule shares
# ---------------------------------------------------------------------------

def test_abstains_without_calibration():
    print('\nabstention without calibration')
    empty = Calibration.from_dict({})
    frames = [[car(200, 360 - i * 14, tid=12)] for i in range(26)]
    for name in ('red_light_jump', 'stop_line_crossing', 'wrong_way'):
        ctx = {'calibration': empty, **RED}
        n = len(run(get_rule(name)({'min_frames': 3}), frames, ctx))
        check(f'{name} abstains with no calibration', n == 0, f'got {n}')


if __name__ == '__main__':
    print('=' * 62)
    print('RULE LOGIC TESTS -- no footage required')
    print('=' * 62)
    test_triple_riding()
    test_red_light_jump()
    test_stop_line_crossing()
    test_abstains_without_calibration()
    print('\n' + '=' * 62)
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
    print('=' * 62)
    sys.exit(1 if FAIL else 0)
