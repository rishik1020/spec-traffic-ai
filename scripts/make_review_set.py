"""
make_review_set.py
==================
Build the evidence set for the EDS validation experiment.

THE PROBLEM THIS SOLVES
The experiment needs ~150 packages that a human will approve some of and reject
others. With the calibration corrected, our footage yields ONE violation -- and
a set where every package is obviously good measures nothing at all. An AUC
computed over fifteen approvals and zero rejections is not a weak result, it is
not a result.

So the set has to span a range of evidence quality ON PURPOSE. That is not a
flaw in the method, it is the method: you cannot measure whether a score
predicts rejection without packages worth rejecting.

HOW THE RANGE IS PRODUCED, AND THE LINE THIS WILL NOT CROSS
Each rule is run several times at different SENSITIVITIES -- how many frames the
condition must hold, how far a vehicle must travel, how far past the threshold
it must be. A strict pass yields few, strong packages. A loose pass yields those
plus marginal ones and outright false positives.

What does NOT vary is the calibration. Lane headings, stop lines and homography
stay exactly as measured, because those are claims about the WORLD and
loosening them would not produce weak evidence, it would produce wrong evidence
of a different kind. Sensitivity is a claim about how eager the rule is, and
varying it is the honest knob.

DEDUPLICATION
The same vehicle is usually caught by several passes. Those are one real event,
not five packages, so they are collapsed on (camera, rule, track) and the
version from the STRICTEST pass that found it is kept -- its best evidence.
What survives from the loose passes is only the events stricter settings
rejected, which is precisely the marginal material the experiment needs.

PROVENANCE
Every package records which pass produced it and at what settings, and the
manifest records the whole sweep. The report must be able to say how the set
was sampled; a validation set whose construction cannot be described is not
evidence of anything.

USAGE
    python scripts/make_review_set.py --source data/videos/kolkata.mp4 \
        --config config/cam_kolkata.yaml
    python scripts/make_review_set.py --source a.mp4 --config a.yaml \
        --source b.mp4 --config b.yaml --target 150 --device cuda
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
# Ordered STRICTEST FIRST. The order is what deduplication uses to decide which
# copy of a repeated event to keep, so it is part of the method, not cosmetic.
#
# Each entry overrides whatever the camera config sets. Rules absent from a
# variant simply run at the camera's own settings.

SWEEP = [
    ('strict', {
        'wrong_way': {'tolerance_deg': 90, 'min_displacement': 60, 'min_frames': 12},
        'red_light_jump': {'min_frames': 8, 'min_progress_px': 45,
                           'min_signal_score': 0.12},
        'stop_line_crossing': {'min_frames': 35, 'min_beyond_px': 25},
        'triple_riding': {'min_frames': 20, 'min_overlap': 0.45},
    }),
    ('default', {}),            # whatever the camera profile declares
    ('relaxed', {
        'wrong_way': {'tolerance_deg': 60, 'min_displacement': 30, 'min_frames': 6,
                      'cooldown_frames': 90},
        'red_light_jump': {'min_frames': 3, 'min_progress_px': 18},
        'stop_line_crossing': {'min_frames': 15, 'min_beyond_px': 8},
        'triple_riding': {'min_frames': 8, 'min_overlap': 0.28,
                          'min_rider_conf': 0.20},
    }),
    ('loose', {
        'wrong_way': {'tolerance_deg': 50, 'min_displacement': 18, 'min_frames': 4,
                      'cooldown_frames': 45},
        'red_light_jump': {'min_frames': 2, 'min_progress_px': 10,
                           'min_signal_score': 0.04},
        'stop_line_crossing': {'min_frames': 8, 'min_beyond_px': 4},
        'triple_riding': {'min_frames': 5, 'min_overlap': 0.22,
                          'min_rider_conf': 0.15},
    }),
]


def variant_config(base: dict, overrides: dict) -> dict:
    """Camera config with rule thresholds overridden. Calibration untouched."""
    cfg = json.loads(json.dumps(base))        # deep copy via round-trip
    rules = cfg.get('rules') or {}
    for name, params in overrides.items():
        if name in rules:                     # only rules this camera enables
            if not isinstance(rules[name], dict):
                rules[name] = {}
            rules[name].update(params)
    cfg['rules'] = rules
    return cfg


def run_variant(source: str, cfg: dict, out_dir: Path, device: str,
                max_frames: int | None) -> int:
    """Run the pipeline once into its own evidence directory."""
    from senti.engine import Engine

    tmp_cfg = out_dir / '_config.yaml'
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

    engine = Engine(config_path=tmp_cfg, evidence_root=out_dir, device=device)
    violations = engine.run(source, max_frames=max_frames)
    return len(violations)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--source', action='append', required=True,
                    help='video; repeat alongside --config for several cameras')
    ap.add_argument('--config', action='append', required=True)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'evidence'))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--target', type=int, default=150,
                    help='how many packages the experiment wants')
    ap.add_argument('--keep-existing', action='store_true',
                    help='add to the output directory instead of replacing it')
    args = ap.parse_args()

    if len(args.source) != len(args.config):
        raise SystemExit('--source and --config must be given in pairs')

    out = Path(args.out)
    if out.is_dir() and not args.keep_existing:
        removed = 0
        for d in out.iterdir():
            if d.is_dir() and (d / 'evidence.json').is_file():
                shutil.rmtree(d)
                removed += 1
        if removed:
            print(f'[set] cleared {removed} existing package(s) from {out}')
    out.mkdir(parents=True, exist_ok=True)

    # (camera, rule, track) -> (strictness rank, folder path, variant)
    best: dict[tuple, tuple] = {}
    manifest = {'sweep': [], 'sources': []}

    work = Path(tempfile.mkdtemp(prefix='reviewset_'))
    try:
        for source, config in zip(args.source, args.config):
            base = yaml.safe_load(Path(config).read_text(encoding='utf-8')) or {}
            enabled = [k for k, v in (base.get('rules') or {}).items() if v is not False]
            print(f'\n[set] {source}')
            print(f'[set]   config {config}, rules {enabled}')
            manifest['sources'].append({'video': source, 'config': config,
                                        'rules': enabled})

            for rank, (name, overrides) in enumerate(SWEEP):
                cfg = variant_config(base, overrides)
                vdir = work / f'{Path(source).stem}_{name}'
                n = run_variant(source, cfg, vdir, args.device, args.max_frames)
                print(f'[set]   {name:8s} -> {n:3d} violation(s)')
                manifest['sweep'].append({
                    'variant': name, 'strictness_rank': rank,
                    'source': source,
                    'overrides': overrides, 'violations': n})

                for folder in sorted(vdir.iterdir()):
                    meta = folder / 'evidence.json'
                    if not meta.is_file():
                        continue
                    rec = json.loads(meta.read_text(encoding='utf-8'))
                    key = (rec.get('camera_id'),
                           (rec.get('violation') or {}).get('rule'),
                           (rec.get('violation') or {}).get('track_id'))
                    # strictest pass wins: it found the same event on the
                    # strongest evidence
                    if key not in best or rank < best[key][0]:
                        best[key] = (rank, folder, name)

        print(f'\n[set] {len(best)} distinct event(s) after deduplication')

        bands = Counter()
        rules = Counter()
        variants = Counter()
        for (cam, rule, tid), (rank, folder, variant) in sorted(best.items()):
            dest = out / f'{folder.name}__{variant}'
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(folder, dest)

            meta = dest / 'evidence.json'
            rec = json.loads(meta.read_text(encoding='utf-8'))
            rec['evidence_id'] = dest.name
            # provenance travels with the package, not just in the manifest
            rec['sweep'] = {
                'variant': variant, 'strictness_rank': rank,
                'overrides': SWEEP[rank][1].get(rule, {}),
                'note': ('Sensitivity varied to span a range of evidence '
                         'quality for the EDS validation experiment. '
                         'Calibration was NOT varied.'),
            }
            meta.write_text(json.dumps(rec, indent=2), encoding='utf-8')

            bands[(rec.get('defensibility') or {}).get('verdict', '?')] += 1
            rules[rule] += 1
            variants[variant] += 1
    finally:
        shutil.rmtree(work, ignore_errors=True)

    total = sum(bands.values())
    print('\n' + '=' * 64)
    print(f'REVIEW SET: {total} package(s) -> {out}')
    print('=' * 64)
    print('  by EDS band :', dict(bands))
    print('  by rule     :', dict(rules))
    print('  by variant  :', dict(variants))

    manifest.update(total=total, bands=dict(bands), rules=dict(rules),
                    variants=dict(variants))
    (out.parent / 'review_set_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\n  manifest -> {out.parent / "review_set_manifest.json"}')

    # -- is this set actually fit for the experiment? -----------------------
    print('\n' + '-' * 64)
    problems = []
    if total < args.target:
        problems.append(f'{total} packages, short of the {args.target} target. '
                        f'Add more footage -- more sweep passes would only '
                        f'produce near-duplicates of the same events.')
    weakest = min(bands.values()) if bands else 0
    if total and weakest / total < 0.15:
        problems.append('One EDS band holds under 15% of the set. A reviewer '
                        'needs enough of each for the comparison to mean '
                        'anything.')
    if len(rules) < 2:
        problems.append(f'Only one rule ({", ".join(rules)}) is represented, so '
                        f'the result speaks for that rule, not for EDS in '
                        f'general. Say so in the write-up.')
    if problems:
        print('BEFORE RUNNING THE EXPERIMENT:')
        for p in problems:
            print('  * ' + p)
    else:
        print('Set looks usable.')
    print('-' * 64)
    print('\nNext:')
    print('  python portal/server.py --blind --shuffle')
    print('  ...review everything, then:')
    print('  python scripts/eds_validate.py')


if __name__ == '__main__':
    main()
