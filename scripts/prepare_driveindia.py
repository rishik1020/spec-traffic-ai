"""
prepare_driveindia.py
=====================
Turn the 11 GB DriveIndia.zip into a Colab-sized subset you can actually upload.

WHAT IT DOES
  1. Extracts the nested zips (train1/2/3, val, test)
  2. Pairs every image with its YOLO label
  3. Samples a subset -- RARE CLASSES FIRST, then fills randomly
  4. Preserves DriveIndia's OFFICIAL train/val/test splits (do not re-split;
     their test set is the benchmark other papers report against)
  5. Writes a clean YOLO tree + data.yaml
  6. Zips it ready for Google Drive

WHY RARE-CLASS-FIRST SAMPLING
  DriveIndia is savagely imbalanced -- Car has ~3,855 boxes in test while
  Rumble strips has 2. Sample 10k images uniformly at random and the rare
  classes essentially vanish, so the model never learns them and your
  per-class chart is full of 0.000 rows. Instead we take every image
  containing a rare class first, then fill the remainder at random. Same
  budget, far better class coverage.

USAGE
    python scripts/prepare_driveindia.py --zip "C:/Users/.../Downloads/DriveIndia.zip"
    python scripts/prepare_driveindia.py --zip <path> --train-images 12000
    python scripts/prepare_driveindia.py --zip <path> --skip-extract   # already extracted
"""

from __future__ import annotations

import argparse
import collections
import random
import shutil
import zipfile
from pathlib import Path

IMG_EXT = {'.jpg', '.jpeg', '.png'}

# Inferred from the DriveIndia paper's class ordering cross-checked against
# observed label frequencies. Ids 24-26 appear in the data but are NOT
# documented in the paper -- verify with the crop cell before publishing.
CLASS_NAMES = {
    0:  'pedestrian',
    1:  'bicycle',
    2:  'car',
    3:  'motorcycle',
    4:  'route_board',
    5:  'bus',
    6:  'commercial_vehicle',
    7:  'truck',
    8:  'traffic_sign',
    9:  'traffic_light',
    10: 'auto_rickshaw',
    11: 'ambulance',
    12: 'construction_vehicle',
    13: 'animal',
    14: 'unmarked_speed_bump',
    15: 'marked_speed_bump',
    16: 'pothole',
    17: 'police_vehicle',
    18: 'tractor',
    19: 'pushcart',
    20: 'temp_traffic_barrier',
    21: 'rumble_strips',
    22: 'traffic_cone',
    23: 'zebra_crossing',
    24: 'undocumented_24',
    25: 'undocumented_25',
    26: 'undocumented_26',
}


def extract_nested(zip_path: Path, work: Path) -> None:
    """DriveIndia.zip contains train1.zip, train2.zip, ... -- unpack both levels."""
    work.mkdir(parents=True, exist_ok=True)
    stage = work / '_zips'
    stage.mkdir(exist_ok=True)

    print(f'[1/5] opening {zip_path.name} ...')
    with zipfile.ZipFile(zip_path) as z:
        inner = [n for n in z.namelist() if n.lower().endswith('.zip')]
        for name in inner:
            out = stage / Path(name).name
            if out.exists():
                print(f'      {out.name} already staged')
                continue
            print(f'      extracting {name} ...')
            with z.open(name) as src, open(out, 'wb') as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)

    for iz in sorted(stage.glob('*.zip')):
        target = work / iz.stem
        if target.exists() and any(target.rglob('*.jpg')):
            print(f'      {iz.stem} already unpacked')
            iz.unlink(missing_ok=True)
            continue
        print(f'      unpacking {iz.name} ...')
        with zipfile.ZipFile(iz) as z:
            z.extractall(target)
        # free the staged copy immediately -- peak disk is the constraint here,
        # the source DriveIndia.zip is untouched and can re-supply it if needed
        iz.unlink(missing_ok=True)
        print(f'      freed {iz.name}')


def index_split(root: Path) -> list[tuple[Path, Path]]:
    """Pair images with labels regardless of the images_2500/labels_2500 naming."""
    pairs = []
    for img in root.rglob('*'):
        if img.suffix.lower() not in IMG_EXT:
            continue
        lbl = None
        for part_i, part in enumerate(img.parts):
            if 'image' in part.lower():
                cand_parts = list(img.parts)
                cand_parts[part_i] = part.lower().replace('image', 'label')
                cand = Path(*cand_parts).with_suffix('.txt')
                if cand.exists():
                    lbl = cand
                    break
        if lbl is None:
            cand = img.with_suffix('.txt')
            lbl = cand if cand.exists() else None
        if lbl:
            pairs.append((img, lbl))
    return pairs


def classes_in(label: Path) -> set[int]:
    out = set()
    try:
        for line in label.read_text().splitlines():
            bits = line.split()
            if bits:
                out.add(int(float(bits[0])))
    except Exception:
        pass
    return out


def sample_rare_first(pairs: list[tuple[Path, Path]], budget: int,
                      seed: int = 42) -> list[tuple[Path, Path]]:
    """Take every image containing a rare class first, then fill at random."""
    if budget >= len(pairs):
        return pairs

    per_image = {p: classes_in(l) for p, l in pairs}
    freq = collections.Counter()
    for cls_set in per_image.values():
        freq.update(cls_set)

    # rarest class first
    rarity = sorted(freq, key=lambda c: freq[c])
    chosen: list[tuple[Path, Path]] = []
    taken = set()

    for cls in rarity:
        if len(chosen) >= budget:
            break
        for img, lbl in pairs:
            if img in taken:
                continue
            if cls in per_image[img]:
                chosen.append((img, lbl))
                taken.add(img)
                if len(chosen) >= budget:
                    break

    rest = [(i, l) for i, l in pairs if i not in taken]
    random.Random(seed).shuffle(rest)
    chosen.extend(rest[:max(0, budget - len(chosen))])
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True, help='path to DriveIndia.zip')
    ap.add_argument('--out', default=None, help='output dir (default: alongside zip)')
    ap.add_argument('--train-images', type=int, default=12000)
    ap.add_argument('--val-images', type=int, default=1500)
    ap.add_argument('--test-images', type=int, default=1500)
    ap.add_argument('--skip-extract', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    zip_path = Path(args.zip).resolve()
    work = zip_path.parent / 'DriveIndia_work'
    out = Path(args.out) if args.out else zip_path.parent / 'driveindia_subset'

    if not args.skip_extract:
        extract_nested(zip_path, work)
    else:
        print('[1/5] skipping extraction')

    # ---- gather each split -------------------------------------------------
    print('[2/5] indexing splits ...')
    splits: dict[str, list[tuple[Path, Path]]] = {'train': [], 'val': [], 'test': []}
    for d in sorted(p for p in work.iterdir() if p.is_dir() and p.name != '_zips'):
        name = d.name.lower()
        key = 'train' if 'train' in name else 'val' if 'val' in name else 'test' if 'test' in name else None
        if key is None:
            continue
        found = index_split(d)
        splits[key].extend(found)
        print(f'      {d.name:12s} -> {key:5s}  {len(found)} pairs')

    for k, v in splits.items():
        if not v:
            print(f'      WARNING: {k} split is empty')

    # ---- subset ------------------------------------------------------------
    print('[3/5] sampling (rare classes first) ...')
    budgets = {'train': args.train_images, 'val': args.val_images, 'test': args.test_images}
    picked = {k: sample_rare_first(v, budgets[k], args.seed) for k, v in splits.items()}

    # ---- write the YOLO tree ----------------------------------------------
    print('[4/5] writing subset ...')
    if out.exists():
        shutil.rmtree(out)

    seen = collections.Counter()
    for split, items in picked.items():
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, out / 'images' / split / img.name)
            shutil.copy2(lbl, out / 'labels' / split / f'{img.stem}.txt')
            seen.update(classes_in(lbl))
        print(f'      {split:5s} {len(items):6d}')

    present = sorted(seen)
    names_block = ''.join(f'  {c}: {CLASS_NAMES.get(c, f"class_{c}")}\n' for c in range(max(present) + 1))
    (out / 'data.yaml').write_text(
        f'path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\n'
        f'nc: {max(present) + 1}\nnames:\n{names_block}',
        encoding='utf-8')

    print('\n      class coverage in subset:')
    for c in present:
        print(f'        {c:2d} {CLASS_NAMES.get(c, "?"):<24} {seen[c]:7d}')

    # ---- zip ---------------------------------------------------------------
    print('[5/5] zipping ...')
    archive = shutil.make_archive(str(out), 'zip', str(out))
    mb = Path(archive).stat().st_size / 1e6
    print(f'\nDONE -> {archive}  ({mb:.0f} MB)')
    print('Upload that zip to Google Drive at  MyDrive/senti/datasets/')


if __name__ == '__main__':
    main()
