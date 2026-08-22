"""
prune_classes.py
================
Drop classes the model can never learn, and remap the rest to a dense id range.

WHY THIS MATTERS FOR YOUR NUMBERS
The DriveIndia subset has 28 classes, but 8 of them have zero or near-zero
support: ambulance, animal and police_vehicle have literally 0 boxes; pothole
has 1, pushcart 4, rumble_strips 22. Those classes will score 0.000 mAP no
matter how good the detector is.

mAP is a MEAN over classes. Leaving 8 dead classes in drags your headline figure
down by roughly 28% for reasons that have nothing to do with model quality, and
makes the number incomparable to DriveIndia's published 78.7% baseline.

Pruning also stops the classification head from allocating capacity to outputs
that can never be correct.

WHAT IT DOES NOT DO
It does not touch your original folder. Images are HARDLINKED into the new
folder, so this costs almost no disk space -- only the rewritten label files.

USAGE
    python scripts/prune_classes.py --data "C:/.../driveindia_subset"
    python scripts/prune_classes.py --data <dir> --min-boxes 50
"""

from __future__ import annotations

import argparse
import collections
import os
import shutil
from pathlib import Path

import yaml

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def count_boxes(ds: Path, split: str) -> collections.Counter:
    c = collections.Counter()
    for t in (ds / 'labels' / split).glob('*.txt'):
        try:
            for line in t.read_text().splitlines():
                bits = line.split()
                if bits:
                    c[int(float(bits[0]))] += 1
        except Exception:
            pass
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default=None, help='default: <data>_pruned')
    ap.add_argument('--min-boxes', type=int, default=25,
                    help='drop classes with fewer than this many TRAIN boxes')
    args = ap.parse_args()

    ds = Path(args.data).resolve()
    out = Path(args.out) if args.out else ds.parent / (ds.name + '_pruned')

    cfg = yaml.safe_load((ds / 'data.yaml').read_text(encoding='utf-8'))
    names = cfg['names']
    names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))

    train_counts = count_boxes(ds, 'train')

    keep = sorted(c for c in names if train_counts.get(c, 0) >= args.min_boxes)
    drop = sorted(c for c in names if c not in keep)

    print(f'threshold: >= {args.min_boxes} train boxes\n')
    print('KEEPING:')
    for c in keep:
        print(f'  {c:2d} {names[c]:<24} {train_counts.get(c, 0):7d}')
    print('\nDROPPING:')
    for c in drop:
        print(f'  {c:2d} {names[c]:<24} {train_counts.get(c, 0):7d}')

    if not drop:
        print('\nnothing to prune.')
        return

    remap = {old: new for new, old in enumerate(keep)}

    if out.exists():
        shutil.rmtree(out)

    stats = collections.Counter()
    for split in ('train', 'val', 'test'):
        src_i, src_l = ds / 'images' / split, ds / 'labels' / split
        if not src_i.is_dir():
            continue
        dst_i, dst_l = out / 'images' / split, out / 'labels' / split
        dst_i.mkdir(parents=True, exist_ok=True)
        dst_l.mkdir(parents=True, exist_ok=True)

        kept_imgs = 0
        for img in src_i.iterdir():
            if img.suffix.lower() not in IMG_EXT:
                continue
            lbl = src_l / f'{img.stem}.txt'
            if not lbl.exists():
                continue

            lines = []
            for line in lbl.read_text().splitlines():
                bits = line.split()
                if len(bits) < 5:
                    continue
                cid = int(float(bits[0]))
                if cid not in remap:
                    continue                       # dropped class -> drop box
                lines.append(' '.join([str(remap[cid])] + bits[1:]))
                stats[remap[cid]] += 1

            # An image whose every box belonged to a dropped class becomes a
            # pure-background image. A few are useful (they teach the model what
            # is NOT a vehicle); thousands would skew training. We keep them --
            # in practice almost every DriveIndia image also contains a car,
            # pedestrian or motorcycle, so this is rare.
            try:
                os.link(img, dst_i / img.name)     # hardlink: no extra disk
            except OSError:
                shutil.copy2(img, dst_i / img.name)
            (dst_l / f'{img.stem}.txt').write_text('\n'.join(lines), encoding='utf-8')
            kept_imgs += 1
        print(f'\n{split:5s} {kept_imgs} images')

    new_names = {remap[c]: names[c] for c in keep}
    (out / 'data.yaml').write_text(yaml.safe_dump({
        'path': str(out),
        'train': 'images/train', 'val': 'images/val', 'test': 'images/test',
        'nc': len(keep),
        'names': {i: new_names[i] for i in sorted(new_names)},
    }, sort_keys=False), encoding='utf-8')

    print(f'\n{len(names)} classes -> {len(keep)} classes')
    print(f'-> {out}')
    print(f'-> {out / "data.yaml"}')


if __name__ == '__main__':
    main()
