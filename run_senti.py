"""
run_senti.py
============
Entry point for the SPEC Traffic AI pipeline.

    python run_senti.py --source clip.mp4
    python run_senti.py --source clip.mp4 --show
    python run_senti.py --source rtsp://10.0.0.5:554/stream1 --config config/cam_hyd01.yaml
    python run_senti.py --list-rules
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description='SPEC Traffic AI')
    ap.add_argument('--source', help='video path, RTSP URL, or webcam index')
    ap.add_argument('--config', default='config/cam_demo.yaml')
    ap.add_argument('--weights', default=None, help='override the config weights')
    ap.add_argument('--evidence', default='data/evidence')
    ap.add_argument('--device', default=None,
                    help='cuda | cpu. Use cpu to stay off a GPU busy training.')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--list-rules', action='store_true')
    args = ap.parse_args()

    from senti.engine import Engine
    from senti.rules.base import RULE_REGISTRY, available_rules

    if args.list_rules:
        print('registered rules:\n')
        for name in available_rules():
            cls = RULE_REGISTRY[name]
            print(f'  {name}')
            print(f'      {cls.description}')
            print(f'      applies_to : {", ".join(cls.applies_to)}')
            print(f'      requires   : {", ".join(cls.requires) or "-"}')
            print(f'      stateful   : {cls.stateful}   min_frames: {cls.min_frames}')
            print(f'      legal      : {cls.mv_act_section or "-"}')
            print()
        return

    if not args.source:
        ap.error('--source is required (or use --list-rules)')

    source = int(args.source) if str(args.source).isdigit() else args.source

    engine = Engine(
        config_path=args.config,
        weights=args.weights,
        evidence_root=args.evidence,
        show=args.show,
        device=args.device,
    )
    violations = engine.run(source, max_frames=args.max_frames)

    if violations:
        print('\nsummary:')
        by_rule: dict[str, int] = {}
        for v in violations:
            by_rule[v.rule_name] = by_rule.get(v.rule_name, 0) + 1
        for name, n in sorted(by_rule.items()):
            print(f'  {name:<20} {n}')
        print(f'\nevidence -> {Path(args.evidence).resolve()}')


if __name__ == '__main__':
    main()
