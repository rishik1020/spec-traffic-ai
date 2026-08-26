"""
simulate_signal.py
==================
Drive the adaptive controller with scripted demand and print what it recommends.

WHY THIS EXISTS
Adaptive timing cannot be demonstrated on the footage we have -- it needs a
signalised junction with two arms in frame, and it needs to be watched for
several minutes to see the plan converge. This feeds the controller demand
readings directly, so the ARITHMETIC can be checked long before the camera is
available, and so a reviewer can see the behaviour that matters:

  * green follows the queue, not the clock
  * a quiet arm still gets its minimum every cycle -- it is never starved
  * the plan converges over several cycles instead of lurching
  * when the junction is over capacity the controller says so rather than
    returning a nonsense cycle length

It reads the same `signal_control:` block a real camera uses, so what you see
here is what the camera would recommend.

USAGE
    python scripts/simulate_signal.py
    python scripts/simulate_signal.py --config config/cam_junction.yaml
    python scripts/simulate_signal.py --scenario oversaturated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from senti.signal import AdaptiveController, ApproachDemand   # noqa: E402

# Scenarios are (queue PCU, flow PCU/hr) per arm, per cycle. Written out rather
# than generated so the expected behaviour is readable next to the output.
SCENARIOS: dict[str, list[dict[str, tuple[float, float]]]] = {

    # The case the feature was asked for: one arm carrying most of the load.
    'imbalanced': [
        {'north': (28, 1500), 'south': (24, 1350), 'east': (4, 260), 'west': (3, 220)},
        {'north': (34, 1700), 'south': (30, 1520), 'east': (5, 280), 'west': (4, 240)},
        {'north': (38, 1810), 'south': (33, 1600), 'east': (4, 250), 'west': (5, 300)},
        {'north': (30, 1550), 'south': (26, 1400), 'east': (6, 320), 'west': (5, 290)},
    ],

    # Equal QUEUES on unequal arms. The single-lane east/west arms carry the
    # same flow as the two-lane north/south ones, so their flow ratio is twice
    # as high and they have genuinely earned more green. Equal queues are not
    # equal demand -- this scenario exists to make that visible.
    'balanced': [
        {'north': (14, 900), 'south': (13, 880), 'east': (13, 870), 'west': (14, 910)},
        {'north': (15, 940), 'south': (14, 900), 'east': (15, 920), 'west': (13, 880)},
    ],

    # Past capacity. Webster's cycle goes to infinity here; the controller must
    # refuse to pretend and switch to draining the longest queue.
    'oversaturated': [
        {'north': (52, 3200), 'south': (48, 3100), 'east': (30, 1650), 'west': (28, 1600)},
        {'north': (60, 3350), 'south': (55, 3250), 'east': (33, 1700), 'west': (31, 1660)},
    ],

    # One arm completely empty. It must STILL be served every cycle.
    'one_empty': [
        {'north': (26, 1500), 'south': (22, 1300), 'east': (0, 0), 'west': (3, 180)},
        {'north': (29, 1620), 'south': (25, 1400), 'east': (0, 0), 'west': (2, 150)},
    ],
}

DEFAULT_SIGNAL_CFG = {
    'phases': [
        {'name': 'north_south', 'approaches': ['north', 'south']},
        {'name': 'east_west', 'approaches': ['east', 'west']},
    ],
    'saturation_flow_pcu_hr': {'north': 3400, 'south': 3400,
                               'east': 1700, 'west': 1700},
    'min_green_s': {'north_south': 15.0, 'east_west': 12.0},
    'update_every_s': 0.0,          # every call is a new cycle, in simulation
    'baseline_greens_s': {'north_south': 30.0, 'east_west': 25.0},
}


def load_cfg(path: str | None) -> dict:
    if not path:
        return dict(DEFAULT_SIGNAL_CFG)
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    sig = cfg.get('signal_control')
    if not sig:
        raise SystemExit(f'{path} has no `signal_control:` block')
    sig = dict(sig)
    # Force a recompute on every scripted cycle; the real interval is a
    # wall-clock property and would make the simulation return None forever.
    sig['update_every_s'] = 0.0
    # The clock window would gate on the real time of day. The scripted demand
    # IS the peak, so drive the demand trigger instead.
    sig.pop('peak_windows', None)
    return sig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--config', default=None,
                    help='camera YAML to take signal_control from')
    ap.add_argument('--scenario', default='all',
                    choices=['all', *SCENARIOS])
    ap.add_argument('--reservice', action='store_true',
                    help='allow serving the busiest arm twice per cycle')
    args = ap.parse_args()

    names = list(SCENARIOS) if args.scenario == 'all' else [args.scenario]

    for name in names:
        cfg = load_cfg(args.config)
        if args.reservice:
            cfg['allow_reservice'] = True
        ctrl = AdaptiveController(cfg)

        print(f'\n{"=" * 78}\nSCENARIO: {name}\n{"=" * 78}')
        t = 0.0
        for cycle_no, arms in enumerate(SCENARIOS[name], start=1):
            demands = {
                arm: ApproachDemand(name=arm, queue_pcu=q, queue_count=int(q),
                                    present_pcu=q, present_count=int(q),
                                    flow_pcu_hr=f, observed_s=120.0,
                                    # the scripted readings ARE a full sample;
                                    # the warm-up guard is a live-camera concern
                                    warming_up=False)
                for arm, (q, f) in arms.items()
            }
            t += 60.0
            plan = ctrl.update(demands, t)
            if plan is None:
                continue

            queues = '  '.join(f'{a}={d.queue_pcu:.0f}' for a, d in demands.items())
            print(f'\ncycle {cycle_no}   queues: {queues}')
            print(f'  -> {plan.summary()}')
            print(f'     order   : {" -> ".join(plan.order)}')
            print(f'     reason  : {plan.reason}')
            for c in plan.caveats:
                print(f'     !        {c}')

    print('\nADVISORY OUTPUT. No signal head is actuated by this codebase.')


if __name__ == '__main__':
    main()
