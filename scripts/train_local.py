"""
train_local.py
==============
Train the Senti Traffic detector on your own machine -- no Colab, no Drive,
no upload. Reads the subset produced by prepare_driveindia.py.

Does everything the Colab notebook does:
  train -> evaluate on the OFFICIAL test split -> training curves ->
  generalisation-gap chart -> per-class breakdown -> results bundle

TUNED FOR 6 GB VRAM (RTX 4050 laptop). The defaults below are deliberately
conservative -- a CUDA OOM three hours into a run is the worst outcome here.

PREREQ -- your torch must be the CUDA build, not CPU:
    pip uninstall -y torch torchvision
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    python -c "import torch; print(torch.cuda.is_available())"

USAGE
    python scripts/train_local.py --data "C:/Users/.../Downloads/driveindia_subset"
    python scripts/train_local.py --data <dir> --model yolo11m.pt --batch 4
    python scripts/train_local.py --data <dir> --eval-only     # charts from an existing run
"""

from __future__ import annotations

import argparse
import collections
import shutil
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='folder containing data.yaml')
    ap.add_argument('--model', default='yolo11s.pt',
                    help='yolo11n/s/m/l/x. On 6 GB stay at s, or m with --batch 4')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8,
                    help='-1 to auto-fit; a fixed small value is safer on 6 GB')
    ap.add_argument('--workers', type=int, default=4,
                    help='set 0 if Windows DataLoader workers misbehave')
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--name', default='driveindia')
    ap.add_argument('--resume', action='store_true',
                    help='continue an interrupted run from last.pt')
    ap.add_argument('--cos-lr', action='store_true', default=True,
                    help='cosine LR decay (default on)')
    ap.add_argument('--save-period', type=int, default=10,
                    help='write a recoverable checkpoint every N epochs')
    ap.add_argument('--eval-only', action='store_true',
                    help='skip training, regenerate charts from an existing run')
    return ap


def prepare_yaml(ds: Path) -> tuple[Path, dict]:
    """Point data.yaml at this machine's absolute path."""
    y = ds / 'data.yaml'
    if not y.exists():
        raise SystemExit(f'no data.yaml in {ds}')
    d = yaml.safe_load(y.read_text(encoding='utf-8'))
    d['path'] = str(ds)
    d['train'], d['val'], d['test'] = 'images/train', 'images/val', 'images/test'
    y.write_text(yaml.safe_dump(d, sort_keys=False), encoding='utf-8')
    return y, d


def class_counts(ds: Path, names: dict) -> dict:
    ids = collections.Counter()
    for t in (ds / 'labels' / 'train').glob('*.txt'):
        try:
            for line in t.read_text().splitlines():
                bits = line.split()
                if bits:
                    ids[int(float(bits[0]))] += 1
        except Exception:
            pass
    return {names.get(i, f'class_{i}'): n for i, n in ids.items()}


# --- charts ----------------------------------------------------------------

def training_curves(run: Path):
    import pandas as pd
    import matplotlib.pyplot as plt

    csv = run / 'results.csv'
    if not csv.exists():
        print('  (no results.csv, skipping curves)')
        return
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()

    def col(name):
        return df[name] if name in df.columns else None

    ep = df['epoch'] if 'epoch' in df else range(len(df))
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Senti Traffic - training diagnostics', fontsize=15, fontweight='bold')

    ax[0, 0].plot(ep, col('train/box_loss'), label='train', lw=2)
    if col('val/box_loss') is not None:
        ax[0, 0].plot(ep, col('val/box_loss'), label='val', lw=2, ls='--')
    ax[0, 0].set_title('Box loss (localisation)'); ax[0, 0].legend()

    ax[0, 1].plot(ep, col('train/cls_loss'), label='train', lw=2, color='tab:orange')
    if col('val/cls_loss') is not None:
        ax[0, 1].plot(ep, col('val/cls_loss'), label='val', lw=2, ls='--', color='tab:red')
    ax[0, 1].set_title('Classification loss'); ax[0, 1].legend()

    ax[1, 0].plot(ep, col('metrics/mAP50(B)'), label='mAP@50', lw=2, color='tab:green')
    ax[1, 0].plot(ep, col('metrics/mAP50-95(B)'), label='mAP@50-95', lw=2, color='tab:olive')
    ax[1, 0].set_title('Validation mAP'); ax[1, 0].legend(); ax[1, 0].set_ylim(0, 1)

    ax[1, 1].plot(ep, col('metrics/precision(B)'), label='precision', lw=2, color='tab:purple')
    ax[1, 1].plot(ep, col('metrics/recall(B)'), label='recall', lw=2, color='tab:brown')
    ax[1, 1].set_title('Precision & recall'); ax[1, 1].legend(); ax[1, 1].set_ylim(0, 1)

    for a in ax.flat:
        a.set_xlabel('epoch'); a.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(run / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  training_curves.png')


def gap_chart(model, data_yaml: Path, run: Path) -> dict:
    import numpy as np
    import matplotlib.pyplot as plt

    scores = {}
    for split in ['train', 'val', 'test']:
        r = model.val(data=str(data_yaml), split=split, verbose=False)
        scores[split] = {'mAP@50': r.box.map50, 'mAP@50-95': r.box.map,
                         'precision': r.box.mp, 'recall': r.box.mr}
        print(f'  {split:6s} mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}')

    keys = ['mAP@50', 'mAP@50-95', 'precision', 'recall']
    x = np.arange(len(keys)); w = .26
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (split, c) in enumerate(zip(['train', 'val', 'test'],
                                       ['tab:blue', 'tab:orange', 'tab:green'])):
        bars = ax.bar(x + (i - 1) * w, [scores[split][k] for k in keys], w,
                      label=split, color=c)
        ax.bar_label(bars, fmt='%.3f', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(keys); ax.set_ylim(0, 1.08)
    ax.set_title('Generalisation gap - same weights, three splits',
                 fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.savefig(run / 'generalisation_gap.png', dpi=150, bbox_inches='tight')
    plt.close()

    gap = scores['train']['mAP@50'] - scores['test']['mAP@50']
    verdict = ('healthy' if gap < .05 else
               'mild overfit - more data or augmentation' if gap < .15 else
               'SEVERE overfit - do not report the training number')
    print(f'  train->test gap {gap:.4f}  ({verdict})')
    return scores


def per_class_chart(model, data_yaml: Path, run: Path, counts: dict):
    import matplotlib.pyplot as plt

    mt = model.val(data=str(data_yaml), split='test', verbose=False)
    rows = []
    for i, name in model.names.items():
        try:
            rows.append((name, float(mt.box.ap50[i]), int(counts.get(name, 0))))
        except Exception:
            pass
    if not rows:
        return
    rows.sort(key=lambda r: r[1])

    names = [r[0] for r in rows]
    aps = [r[1] for r in rows]
    sup = [r[2] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(15, max(4, len(rows) * .38)), sharey=True)
    cols = ['tab:red' if a < .3 else 'tab:orange' if a < .6 else 'tab:green' for a in aps]
    b1 = ax[0].barh(names, aps, color=cols)
    ax[0].bar_label(b1, fmt='%.3f', fontsize=8, padding=2)
    ax[0].set_xlim(0, 1.12); ax[0].set_xlabel('mAP@50 (test)')
    ax[0].set_title('Per-class accuracy', fontweight='bold')
    ax[0].axvline(mt.box.map50, ls='--', color='k', alpha=.6,
                  label=f'overall {mt.box.map50:.3f}')
    ax[0].legend(loc='lower right'); ax[0].grid(axis='x', alpha=.3)

    b2 = ax[1].barh(names, sup, color='tab:blue')
    ax[1].bar_label(b2, fmt='%d', fontsize=8, padding=2)
    ax[1].set_xlabel('training boxes')
    ax[1].set_title('Class support (imbalance)', fontweight='bold')
    ax[1].grid(axis='x', alpha=.3)

    plt.tight_layout()
    plt.savefig(run / 'per_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  per_class.png')

    weak = [(n, s) for n, a, s in rows if a < .5]
    if weak:
        print('\n  under 0.5 mAP@50:')
        for n, s in weak:
            print(f'    {n:<26} {s:6d} boxes')


# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            'CUDA not available -- you are on the CPU build of torch.\n'
            '  pip uninstall -y torch torchvision\n'
            '  pip install torch torchvision --index-url '
            'https://download.pytorch.org/whl/cu126')

    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU    : {gpu}  ({vram:.1f} GB)')
    print(f'model  : {args.model}   batch={args.batch}  imgsz={args.imgsz}')

    if vram < 8 and args.model.startswith(('yolo11m', 'yolo11l', 'yolo11x')) and args.batch > 4:
        print('WARNING: >6 GB model on a small card. If it OOMs, use --batch 4.')

    ds = Path(args.data).resolve()
    data_yaml, cfg = prepare_yaml(ds)

    names = cfg['names']
    names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    counts = class_counts(ds, names)

    for split in ['train', 'val', 'test']:
        n = len(list((ds / 'images' / split).glob('*')))
        print(f'  {split:5s} {n:6d} images')
    print(f'  {cfg["nc"]} classes\n')

    out = PROJECT_ROOT / 'runs'
    out.mkdir(parents=True, exist_ok=True)
    run = out / args.name

    from ultralytics import YOLO

    if not args.eval_only:
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=0,
            workers=args.workers,
            cache=False,
            patience=args.patience,
            seed=args.seed,
            project=str(out),
            name=args.name,
            exist_ok=True,
            resume=args.resume,

            # ---- schedule ----------------------------------------------
            cos_lr=args.cos_lr,          # cosine decay; better final epochs
            save_period=args.save_period,  # recoverable checkpoints -- a run
                                           # that dies at epoch 40 should not
                                           # cost 40 epochs

            # ---- augmentation, tuned for INDIAN conditions --------------
            # Photometric: Indian footage swings between harsh midday glare,
            # deep shade under flyovers, dust haze and night high-beam. Wide
            # value/saturation jitter is the cheapest robustness we can buy.
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.5,                   # raised from 0.4 default

            # Geometric: cameras are pole-mounted at varying tilt, so a little
            # rotation and scale helps. Keep translate modest -- the subject
            # must stay in frame for evidence to be meaningful.
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,

            # Horizontal flip REDUCED from the 0.5 default. India drives on the
            # left, and DriveIndia includes route_board and traffic_sign, whose
            # text mirrors under a flip. Some flipping still helps generalise
            # vehicle appearance, so we damp rather than disable.
            fliplr=0.25,
            flipud=0.0,

            # Mosaic stitches 4 images together -- the single biggest win for
            # SMALL objects, and this dataset is almost entirely small boxes
            # (traffic lights, distant two-wheelers). close_mosaic turns it off
            # for the final epochs so the model finishes on real layouts.
            mosaic=1.0,
            close_mosaic=10,

            # copy_paste duplicates rare-class instances into other images --
            # directly targets our imbalance (pedestrian 18,880 boxes vs
            # tractor 77).
            copy_paste=0.3,
            mixup=0.0,                   # tends to hurt detection
        )

    best = run / 'weights' / 'best.pt'
    if not best.exists():
        raise SystemExit(f'no weights at {best}')

    print('\n=== evaluating on the OFFICIAL test split ===')
    m = YOLO(str(best))
    mt = m.val(data=str(data_yaml), split='test')
    print(f'\nmAP@50     {mt.box.map50:.4f}')
    print(f'mAP@50-95  {mt.box.map:.4f}')
    print(f'precision  {mt.box.mp:.4f}')
    print(f'recall     {mt.box.mr:.4f}\n')

    print('=== charts ===')
    training_curves(run)
    gap_chart(m, data_yaml, run)
    per_class_chart(m, data_yaml, run, counts)

    # bundle for the report
    bundle = PROJECT_ROOT / 'results' / args.name
    bundle.mkdir(parents=True, exist_ok=True)
    for f in ['training_curves.png', 'generalisation_gap.png', 'per_class.png',
              'confusion_matrix_normalized.png', 'results.png', 'PR_curve.png',
              'F1_curve.png', 'labels.jpg', 'results.csv']:
        if (run / f).exists():
            shutil.copy2(run / f, bundle / f)
    shutil.copy2(best, bundle / 'best.pt')
    print(f'\nbundle -> {bundle}')
    print(f'weights -> {best}')


if __name__ == '__main__':
    # required on Windows: DataLoader workers re-import this module
    main()
