"""
make_figures.py
===============
Produce the report figures into docs/figures/.

WHAT IT MAKES
    fig1_architecture.png  drawn here: one perception pass, two consumers
    fig2_domain_gap.png    MEASURED here: both detectors run on one real frame,
                           boxes drawn, counts taken from the inference output
    fig3_generalisation_gap.png  copied from results/driveindia/
    fig4_per_class.png            copied from results/driveindia/
    fig6_signal_split.png         copied from results/driveindia/

Figure 5 is a screenshot of the running officer portal and cannot be produced
without a browser, so it is left for manual capture.

ON FIGURE 2
The report claims a viewpoint domain gap between the DriveIndia fine-tune and
the UVH-26 model on CCTV footage. This script does not draw that claim -- it
runs both models and prints the counts it actually obtains, so the figure and
the number in the report come from the same source. If the printed counts
differ from the report, the report is what is wrong.

USAGE
    python scripts/make_figures.py
    python scripts/make_figures.py --frame 140 --device cuda
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'docs' / 'figures'

INK = '#1a1a1a'
MUTED = '#6b6b6b'
ACCENT = '#0b5394'
WARM = '#9c4221'


def fig1_architecture() -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis('off')

    def box(x, y, w, h, title, sub='', edge=INK, fill='#ffffff', lw=1.3):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.6,rounding_size=1.4',
            linewidth=lw, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h / 2 + (1.5 if sub else 0), title,
                ha='center', va='center', fontsize=9.5, color=INK,
                fontweight='bold', family='serif')
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.6, sub, ha='center', va='center',
                    fontsize=7.8, color=MUTED, family='serif')

    def arrow(x1, y1, x2, y2, label='', dashed=False):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=11,
            linewidth=1.1, color=MUTED,
            linestyle='--' if dashed else '-', shrinkA=1, shrinkB=1))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.4, label, ha='center',
                    va='bottom', fontsize=7.2, color=MUTED, family='serif',
                    style='italic')

    # shared perception pass
    box(2, 22, 15, 11, 'Video source', 'file or RTSP', edge=MUTED)
    box(21, 22, 23, 11, 'Perception', 'YOLO11 + ByteTrack',
        edge=ACCENT, fill='#f4f8fc', lw=1.8)
    arrow(17, 27.5, 21, 27.5)

    ax.text(32.5, 36.5, 'ONE PASS', ha='center', fontsize=7.6, color=ACCENT,
            family='serif', fontweight='bold')

    # consumer A: enforcement
    box(47, 36, 18, 10, 'Rule engine', 'temporal + abstention', edge=INK)
    box(69, 36, 13, 10, 'EDS', 'triage 0-100', edge=INK)
    box(86, 36, 12, 10, 'Portal', 'officer', edge=WARM, fill='#fdf6f3')
    arrow(44, 29.5, 47, 41)
    arrow(65, 41, 69, 41)
    arrow(82, 41, 86, 41)

    # consumer B: signal timing
    box(47, 8, 18, 10, 'Demand meter', 'queue in PCU', edge=INK)
    box(69, 8, 29, 10, 'Webster controller',
        'C = (1.5L + 5) / (1 - Y)   advisory', edge=INK)
    arrow(44, 25.5, 47, 13)
    arrow(65, 13, 69, 13)

    ax.text(99, 31.5, 'enforcement', ha='right', fontsize=7.4, color=MUTED,
            family='serif', style='italic')
    ax.text(99, 3.5, 'traffic management', ha='right', fontsize=7.4,
            color=MUTED, family='serif', style='italic')

    # the point of the diagram
    ax.plot([45.5, 45.5], [6, 48], color='#d0d0d0', linewidth=0.9, linestyle=':')
    ax.text(45.5, 50.2, 'two consumers, no second inference',
            ha='center', fontsize=7.6, color=INK, family='serif')

    p = OUT / 'fig1_architecture.png'
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return p


# Table 1 of the report: the official evaluation of the DriveIndia fine-tune.
SPLIT_METRICS = {
    'Train':      {'mAP@50': 0.850, 'mAP@50-95': 0.752,
                   'Precision': 0.832, 'Recall': 0.763},
    'Validation': {'mAP@50': 0.716, 'mAP@50-95': 0.598,
                   'Precision': 0.721, 'Recall': 0.679},
    'Test':       {'mAP@50': 0.703, 'mAP@50-95': 0.606,
                   'Precision': 0.620, 'Recall': 0.671},
}


def fig3_generalisation_gap() -> Path:
    """Redrawn from the reported metrics so it prints at the required DPI."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = list(next(iter(SPLIT_METRICS.values())).keys())
    splits = list(SPLIT_METRICS)
    x = np.arange(len(metrics))
    width = 0.26
    shades = ['#c7d9ea', '#6f9fcb', ACCENT]

    fig, ax = plt.subplots(figsize=(8.0, 3.9))
    for i, (split, shade) in enumerate(zip(splits, shades)):
        vals = [SPLIT_METRICS[split][m] for m in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=split,
                      color=shade, edgecolor='white', linewidth=0.8)
        ax.bar_label(bars, fmt='%.3f', fontsize=7.4, padding=2,
                     color=INK, family='serif')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, family='serif', fontsize=9.5, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('score', family='serif', fontsize=9.5, color=INK)
    ax.legend(frameon=False, prop={'family': 'serif', 'size': 9})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    ax.grid(axis='y', color='#e8e8e8', linewidth=0.8)
    ax.set_axisbelow(True)

    gap = (SPLIT_METRICS['Train']['mAP@50']
           - SPLIT_METRICS['Test']['mAP@50'])
    ax.set_title('Train-to-test generalisation gap of '
                 + format(gap, '.3f') + ' on mAP@50',
                 family='serif', fontsize=10, color=INK, pad=10)

    p = OUT / 'fig3_generalisation_gap.png'
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return p


def fig2_domain_gap(frame_idx: int,
                    device: str) -> tuple[Path, int, int, int, int]:
    """Run BOTH detectors on one real frame. Counts come from inference."""
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    src = ROOT / 'data' / 'videos' / 'kolkata.mp4'
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f'could not read frame {frame_idx} from {src}')

    models = [
        ('DriveIndia fine-tune', ROOT / 'results' / 'driveindia' / 'best.pt',
         WARM),
        ('UVH-26 (CCTV)', ROOT / 'models' / 'uvh26' / 'UVH-26-MV-YOLOv11-S.pt',
         ACCENT),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.6))
    counts: list[int] = []
    buses_seen: list[int] = []
    for ax, (label, weights, colour) in zip(axes, models):
        model = YOLO(str(weights))
        res = model.predict(frame, imgsz=1280, conf=0.25, verbose=False,
                            device=device)[0]
        canvas = frame.copy()
        n = buses = 0
        bgr = tuple(int(colour.lstrip('#')[i:i + 2], 16) for i in (4, 2, 0))
        for b in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            name = res.names[int(b.cls)]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr, 2)
            # a bus label on this scene is the domain gap made visible, so it
            # is drawn in full while the rest stay unlabelled for legibility
            if name == 'bus':
                buses += 1
                cv2.putText(canvas, 'bus ' + format(float(b.conf), '.2f'),
                            (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 255), 2, cv2.LINE_AA)
            n += 1
        counts.append(n)
        buses_seen.append(buses)
        extra = ', ' + str(buses) + ' labelled bus' if buses else ''
        ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{label}\n{n} detections{extra}', fontsize=10, family='serif',
                     color=INK)
        ax.axis('off')

    fig.subplots_adjust(wspace=0.02)
    fig.tight_layout(pad=0.4)
    p = OUT / 'fig2_domain_gap.png'
    fig.savefig(p, dpi=380, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return p, counts[0], counts[1], buses_seen[0], buses_seen[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', type=int, default=217)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--skip-gap', action='store_true')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    print(fig1_architecture().name)

    if not args.skip_gap:
        p, a, b, ba, bb = fig2_domain_gap(args.frame, args.device)
        print(f'{p.name}   MEASURED at imgsz=1280 conf=0.25, '
              f'frame {args.frame}:')
        print(f'    DriveIndia {a} detections ({ba} labelled bus)')
        print(f'    UVH-26     {b} detections ({bb} labelled bus)')

    print(fig3_generalisation_gap().name)

    copies = {
        'fig4_per_class.png': 'per_class.png',
        'fig6_signal_split.png': 'signal_split.png',
    }
    for dst, src in copies.items():
        s = ROOT / 'results' / 'driveindia' / src
        if s.exists():
            shutil.copy2(s, OUT / dst)
            print(dst)
        else:
            print(f'{dst}  MISSING SOURCE {s}')

    print('\nFigure 5 (officer portal) needs a screenshot of the running '
          'portal; save it as docs/figures/fig5_portal.png')


if __name__ == '__main__':
    main()
