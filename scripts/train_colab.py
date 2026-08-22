# %% [markdown]
# # Senti Traffic — detector training (Google Colab)
#
# Upload one or more dataset zips; this notebook does the rest.
#
# * Accepts **YOLO** or **Pascal VOC XML**, nested or flat — auto-detected
# * Merges multiple datasets **by class name**, not by class id (see Cell 5)
# * Picks the largest YOLO11 model your GPU can actually train
# * Reports **held-out TEST** mAP, not training accuracy
#
# **Runtime → Change runtime type → T4 GPU** before running anything.

# %%
# --- Cell 1: GPU + install ---------------------------------------------------
!nvidia-smi

%pip install -q ultralytics
import ultralytics; ultralytics.checks()

# %%
# --- Cell 2: pick the best model this GPU can handle -------------------------
import torch

if not torch.cuda.is_available():
    raise SystemExit('No GPU. Runtime -> Change runtime type -> T4 GPU')

GPU  = torch.cuda.get_device_name(0)
VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9

# Bigger = more accurate, slower, more VRAM. These thresholds leave headroom
# for activations at imgsz=640; going larger will OOM mid-run, which on Colab
# means losing the session.
if   VRAM >= 38: MODEL, IMGSZ = 'yolo11x.pt', 640   # A100
elif VRAM >= 20: MODEL, IMGSZ = 'yolo11l.pt', 640   # L4
elif VRAM >= 14: MODEL, IMGSZ = 'yolo11m.pt', 640   # T4  <- Colab free
else:            MODEL, IMGSZ = 'yolo11s.pt', 640

print(f'GPU        : {GPU}  ({VRAM:.1f} GB)')
print(f'model      : {MODEL}')

# --- knobs -------------------------------------------------------------------
EPOCHS     = 60
MAX_IMAGES = None    # e.g. 10000 to cap the dataset; None = use everything
SEED       = 42
# -----------------------------------------------------------------------------

# %%
# --- Cell 3: get the data in -------------------------------------------------
# Handles all three ways data can arrive:
#   A) zips in Google Drive at  MyDrive/senti/datasets/*.zip   <- BEST for >100 MB
#   B) zips via the upload widget                              <- ok under 100 MB
#   C) loose image/label files already dumped into /content    <- salvages a
#                                                                 mis-aimed upload

from pathlib import Path
import shutil, zipfile

WORK = Path('/content/extracted')
RAW  = Path('/content/raw')
for d in (WORK, RAW):
    d.mkdir(parents=True, exist_ok=True)

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}
LBL_EXT = {'.xml', '.txt'}

USE_DRIVE  = True    # mount Drive and pull zips from MyDrive/senti/datasets
USE_WIDGET = False   # set True to also get the browser upload prompt

# --- A) Drive ----------------------------------------------------------------
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        src = Path('/content/drive/MyDrive/senti/datasets')
        if src.is_dir():
            for z in src.glob('*.zip'):
                shutil.copy2(z, RAW / z.name)
                print(f'from Drive: {z.name}  ({z.stat().st_size/1e6:.0f} MB)')
        else:
            print(f'(no {src} yet — create it and drop your zips there)')
    except Exception as e:
        print('drive skipped:', e)

# --- B) widget ---------------------------------------------------------------
if USE_WIDGET:
    from google.colab import files
    print('\nSelect your dataset ZIP (not the extracted files):')
    for name in files.upload():
        if name.lower().endswith('.zip'):
            shutil.move(name, RAW / name)

# --- C) sweep anything loose in /content -------------------------------------
loose = WORK / 'uploaded'
loose.mkdir(parents=True, exist_ok=True)
n_loose = 0
for p in Path('/content').iterdir():
    if not p.is_file():
        continue
    ext = p.suffix.lower()
    if ext == '.zip':
        shutil.move(str(p), RAW / p.name)
    elif ext in IMG_EXT | LBL_EXT:
        shutil.move(str(p), loose / p.name)
        n_loose += 1
if not any(loose.iterdir()):
    loose.rmdir()
elif n_loose:
    print(f'swept {n_loose} loose files from /content')

# --- extract every zip -------------------------------------------------------
for z in sorted(RAW.glob('*.zip')):
    out = WORK / z.stem.replace(' ', '_')
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(out)
    n_img = sum(1 for p in out.rglob('*') if p.suffix.lower() in IMG_EXT)
    n_txt = sum(1 for _ in out.rglob('*.txt'))
    n_xml = sum(1 for _ in out.rglob('*.xml'))
    print(f'{z.name:38s} images={n_img:6d}  yolo_txt={n_txt:6d}  voc_xml={n_xml:6d}')

imgs = [p for p in WORK.rglob('*') if p.suffix.lower() in IMG_EXT]
lbls = [p for p in WORK.rglob('*') if p.suffix.lower() in LBL_EXT]
print(f'\nTOTAL  images={len(imgs)}   labels={len(lbls)}')
assert imgs, 'no images found — check the paths above'

# %%
# --- Cell 5: detect format, convert, merge by NAME ---------------------------
#
# WHY MERGE BY NAME: two YOLO datasets both use ids 0,1,2... but dataset A's
# "0" may be `car` while dataset B's "0" is `auto`. Concatenating them silently
# corrupts every label. We resolve every box to a class *string* first, then
# assign fresh global ids. This is the only safe way to combine datasets.

import xml.etree.ElementTree as ET
import random, yaml

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def swap_dir(path: Path, frm, to, suffix) -> Path | None:
    """images/x.jpg -> labels/x.txt, JPEGImages/x.jpg -> Annotations/x.xml"""
    parts = list(path.parts)
    for i, part in enumerate(parts):
        if part in frm:
            parts[i] = to
            cand = Path(*parts).with_suffix(suffix)
            if cand.exists():
                return cand
    return None


def find_label(img: Path):
    """Return ('yolo', path) or ('voc', path) or None."""
    c = swap_dir(img, {'images', 'JPEGImages'}, 'labels', '.txt') or \
        (img.with_suffix('.txt') if img.with_suffix('.txt').exists() else None)
    if c: return 'yolo', c
    c = swap_dir(img, {'images', 'JPEGImages'}, 'Annotations', '.xml') or \
        (img.with_suffix('.xml') if img.with_suffix('.xml').exists() else None)
    if c: return 'voc', c
    return None


def yolo_names_for(img: Path) -> dict | None:
    """Walk up from an image looking for the dataset's own data.yaml."""
    for parent in list(img.parents)[:6]:
        for y in list(parent.glob('*.yaml')) + list(parent.glob('*.yml')):
            try:
                d = yaml.safe_load(y.read_text())
                if isinstance(d, dict) and d.get('names'):
                    names = d['names']
                    if isinstance(names, list):
                        names = {i: n for i, n in enumerate(names)}
                    return {int(k): str(v) for k, v in names.items()}
            except Exception:
                pass
    return None


def read_voc(xml_path: Path):
    """-> (w, h, [(label, xmin, ymin, xmax, ymax)])"""
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    w = int(float(size.findtext('width', '0')))
    h = int(float(size.findtext('height', '0')))
    boxes = []
    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip()
        bb = obj.find('bndbox')
        if not name or bb is None:
            continue
        boxes.append((name,
                      float(bb.findtext('xmin', '0')), float(bb.findtext('ymin', '0')),
                      float(bb.findtext('xmax', '0')), float(bb.findtext('ymax', '0'))))
    return w, h, boxes


# ---- pass 1: collect every (image, boxes-as-names) --------------------------
from PIL import Image as PILImage

records = []          # (img_path, [(name, xc, yc, bw, bh) normalised])
per_dataset = {}

for ds in sorted(p for p in WORK.iterdir() if p.is_dir()):
    kept = 0
    names_cache = None
    for img in ds.rglob('*'):
        if img.suffix.lower() not in IMG_EXT:
            continue
        found = find_label(img)
        if not found:
            continue
        kind, lbl = found
        boxes = []

        if kind == 'yolo':
            if names_cache is None:
                names_cache = yolo_names_for(img) or {}
            for line in lbl.read_text().split('\n'):
                bits = line.split()
                if len(bits) < 5:
                    continue
                cid = int(float(bits[0]))
                boxes.append((names_cache.get(cid, f'{ds.name}_class_{cid}'),
                              *(float(v) for v in bits[1:5])))
        else:  # voc -> normalise here
            try:
                w, h, raw = read_voc(lbl)
                if w <= 0 or h <= 0:
                    w, h = PILImage.open(img).size
            except Exception:
                continue
            for name, x1, y1, x2, y2 in raw:
                x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
                y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
                bw, bh = x2 - x1, y2 - y1
                if bw <= 1 or bh <= 1:
                    continue
                boxes.append((name, (x1+x2)/2/w, (y1+y2)/2/h, bw/w, bh/h))

        if boxes:
            records.append((img, boxes))
            kept += 1
    per_dataset[ds.name] = kept
    print(f'{ds.name:40s} {kept:6d} labelled images')

assert records, 'no labelled images found — paste Cell 4 output and I will fix the matcher'

# ---- build the union taxonomy ----------------------------------------------
all_names = sorted({n for _, boxes in records for n, *_ in boxes})
NAME2ID = {n: i for i, n in enumerate(all_names)}

print(f'\n{len(records)} labelled images, {len(all_names)} classes:')
counts = {}
for _, boxes in records:
    for n, *_ in boxes:
        counts[n] = counts.get(n, 0) + 1
for n in sorted(counts, key=counts.get, reverse=True):
    print(f'  {n:<28} {counts[n]:6d}')

if len(records) < 500:
    print('\n*** WARNING: fewer than 500 images. This will not train a usable '
          'model — it only proves the pipeline runs. ***')

# %%
# --- Cell 6: write splits + data.yaml ---------------------------------------
random.seed(SEED)
random.shuffle(records)
if MAX_IMAGES:
    records = records[:MAX_IMAGES]

n = len(records)
a, b = int(n * .8), int(n * .9)
splits = {'train': records[:a], 'val': records[a:b], 'test': records[b:]}

DS = Path('/content/ds')
if DS.exists(): shutil.rmtree(DS)

for split, items in splits.items():
    (DS/'images'/split).mkdir(parents=True, exist_ok=True)
    (DS/'labels'/split).mkdir(parents=True, exist_ok=True)
    for i, (img, boxes) in enumerate(items):
        stem = f'{i:06d}_{img.stem}'[:80]
        shutil.copy2(img, DS/'images'/split/f'{stem}{img.suffix.lower()}')
        (DS/'labels'/split/f'{stem}.txt').write_text('\n'.join(
            f'{NAME2ID[nm]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}'
            for nm, xc, yc, bw, bh in boxes))
    print(f'{split:5s} {len(items):6d}')

DATA_YAML = DS/'data.yaml'
DATA_YAML.write_text(yaml.safe_dump({
    'path': str(DS), 'train': 'images/train',
    'val': 'images/val', 'test': 'images/test',
    'nc': len(all_names),
    'names': {i: n for n, i in sorted(NAME2ID.items(), key=lambda kv: kv[1])},
}, sort_keys=False))
print('\n' + DATA_YAML.read_text())

# %%
# --- Cell 7: train -----------------------------------------------------------
from ultralytics import YOLO

OUT = Path('/content/drive/MyDrive/senti/runs') \
      if Path('/content/drive/MyDrive').exists() else Path('/content/runs')
OUT.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL)          # COCO weights — fine-tune, never from scratch

model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=-1,        # auto-fit to VRAM
    device=0,
    workers=2,       # Colab free = 2 CPU cores
    cache=False,     # caching to RAM OOMs a ~12 GB runtime
    patience=15,     # early stop when val plateaus
    seed=SEED,
    project=str(OUT),
    name='senti',
    exist_ok=True,
)
print('weights ->', OUT/'senti'/'weights'/'best.pt')

# %%
# --- Cell 8: evaluate on the held-out TEST split ----------------------------
# NOT val. val was used for early stopping, so it is no longer unseen data.
# Reporting test mAP is the methodology claim — most student papers report
# training accuracy and inflate their numbers by 15-25 points.

best = OUT/'senti'/'weights'/'best.pt'
m = YOLO(str(best))

mt = m.val(data=str(DATA_YAML), split='test')
print(f'\nmAP@50     {mt.box.map50:.4f}')
print(f'mAP@50-95  {mt.box.map:.4f}')
print(f'precision  {mt.box.mp:.4f}')
print(f'recall     {mt.box.mr:.4f}\n')

for i, name in m.names.items():
    try:
        print(f'  {name:<28} mAP50={mt.box.ap50[i]:.3f}')
    except Exception:
        pass

# %%
# --- Cell 9: training curves -------------------------------------------------
# Ultralytics logs every epoch to results.csv. These four panels are the ones
# that actually tell you something.
#
# READING THEM:
#   train loss falling + val loss RISING  = overfitting, stop earlier / more data
#   both flat from the start              = learning rate or data problem
#   mAP still climbing at the last epoch  = undertrained, raise EPOCHS

import pandas as pd
import matplotlib.pyplot as plt

RUN = OUT/'senti'
df = pd.read_csv(RUN/'results.csv')
df.columns = df.columns.str.strip()      # ultralytics pads these with spaces


def col(*cands):
    for c in cands:
        if c in df.columns:
            return df[c]
    return None


epochs = df['epoch'] if 'epoch' in df else range(len(df))
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Senti Traffic — training diagnostics', fontsize=15, fontweight='bold')

# 1. box loss: the overfitting check
ax[0,0].plot(epochs, col('train/box_loss'), label='train', lw=2)
if col('val/box_loss') is not None:
    ax[0,0].plot(epochs, col('val/box_loss'), label='val', lw=2, ls='--')
ax[0,0].set_title('Box loss  (localisation)'); ax[0,0].legend()

# 2. classification loss
ax[0,1].plot(epochs, col('train/cls_loss'), label='train', lw=2, color='tab:orange')
if col('val/cls_loss') is not None:
    ax[0,1].plot(epochs, col('val/cls_loss'), label='val', lw=2, ls='--', color='tab:red')
ax[0,1].set_title('Classification loss'); ax[0,1].legend()

# 3. mAP — the number that matters
ax[1,0].plot(epochs, col('metrics/mAP50(B)'), label='mAP@50', lw=2, color='tab:green')
ax[1,0].plot(epochs, col('metrics/mAP50-95(B)'), label='mAP@50-95', lw=2, color='tab:olive')
ax[1,0].set_title('Validation mAP'); ax[1,0].legend(); ax[1,0].set_ylim(0, 1)

# 4. precision / recall trade-off
ax[1,1].plot(epochs, col('metrics/precision(B)'), label='precision', lw=2, color='tab:purple')
ax[1,1].plot(epochs, col('metrics/recall(B)'), label='recall', lw=2, color='tab:brown')
ax[1,1].set_title('Precision & recall'); ax[1,1].legend(); ax[1,1].set_ylim(0, 1)

for a in ax.flat:
    a.set_xlabel('epoch'); a.grid(alpha=.3)

plt.tight_layout()
plt.savefig(RUN/'training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

best_ep = int(col('metrics/mAP50-95(B)').idxmax())
print(f'best epoch: {best_ep}  (mAP50-95 = {col("metrics/mAP50-95(B)").max():.4f})')
print(f'trained for {len(df)} epochs')

# %%
# --- Cell 10: the generalisation-gap chart ----------------------------------
# THE chart for your report. Evaluates the SAME weights on all three splits.
#
# A small train->test drop means the model generalises. A large one means it
# memorised. Papers that report only training accuracy are hiding this gap —
# showing it is the point.

import numpy as np

scores = {}
for split in ['train', 'val', 'test']:
    r = m.val(data=str(DATA_YAML), split=split, verbose=False)
    scores[split] = {'mAP@50': r.box.map50, 'mAP@50-95': r.box.map,
                     'precision': r.box.mp, 'recall': r.box.mr}
    print(f'{split:6s} mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}')

metrics_list = ['mAP@50', 'mAP@50-95', 'precision', 'recall']
x = np.arange(len(metrics_list)); w = 0.26

fig, ax = plt.subplots(figsize=(11, 5.5))
for i, (split, colr) in enumerate(zip(['train','val','test'],
                                      ['tab:blue','tab:orange','tab:green'])):
    vals = [scores[split][k] for k in metrics_list]
    bars = ax.bar(x + (i-1)*w, vals, w, label=split, color=colr)
    ax.bar_label(bars, fmt='%.3f', fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(metrics_list)
ax.set_ylim(0, 1.08); ax.set_ylabel('score')
ax.set_title('Generalisation gap — same weights, three splits',
             fontsize=13, fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=.3)
plt.tight_layout()
plt.savefig(RUN/'generalisation_gap.png', dpi=150, bbox_inches='tight')
plt.show()

gap = scores['train']['mAP@50'] - scores['test']['mAP@50']
print(f'\ntrain->test mAP@50 gap: {gap:.4f}')
print('  < 0.05  healthy' if gap < .05 else
      '  0.05-0.15  mild overfit — more data or augmentation' if gap < .15 else
      '  > 0.15  SEVERE overfit — do not report the training number')

# %%
# --- Cell 11: per-class performance on TEST ---------------------------------
# Aggregate mAP hides everything. A model can score 0.85 overall while being
# useless on auto-rickshaws — which for this project is the class that matters.

mt = m.val(data=str(DATA_YAML), split='test', verbose=False)

rows = []
for i, name in m.names.items():
    try:
        rows.append((name, float(mt.box.ap50[i]), int(counts.get(name, 0))))
    except Exception:
        pass
rows.sort(key=lambda r: r[1])

names_  = [r[0] for r in rows]
aps     = [r[1] for r in rows]
support = [r[2] for r in rows]

fig, ax = plt.subplots(1, 2, figsize=(15, max(4, len(rows)*.42)), sharey=True)

colors = ['tab:red' if a < .3 else 'tab:orange' if a < .6 else 'tab:green' for a in aps]
b1 = ax[0].barh(names_, aps, color=colors)
ax[0].bar_label(b1, fmt='%.3f', fontsize=9, padding=2)
ax[0].set_xlim(0, 1.12); ax[0].set_xlabel('mAP@50 (test)')
ax[0].set_title('Per-class accuracy', fontweight='bold')
ax[0].axvline(mt.box.map50, ls='--', color='k', alpha=.6,
              label=f'overall {mt.box.map50:.3f}')
ax[0].legend(loc='lower right'); ax[0].grid(axis='x', alpha=.3)

b2 = ax[1].barh(names_, support, color='tab:blue')
ax[1].bar_label(b2, fmt='%d', fontsize=9, padding=2)
ax[1].set_xlabel('training boxes')
ax[1].set_title('Class support (imbalance)', fontweight='bold')
ax[1].grid(axis='x', alpha=.3)

plt.tight_layout()
plt.savefig(RUN/'per_class.png', dpi=150, bbox_inches='tight')
plt.show()

weak = [n for n, a, _ in rows if a < .5]
if weak:
    print('under 0.5 mAP@50 — needs more data or is genuinely hard:')
    for n in weak:
        print(f'  {n:<28} {dict((r[0], r[2]) for r in rows)[n]:5d} boxes')

# %%
# --- Cell 12: ultralytics' own plots ----------------------------------------
# Generated automatically during training. The confusion matrix is the useful
# one: it shows WHICH classes get mistaken for each other (expect auto <-> car
# confusion, which is exactly the Indian-classes problem).

from IPython.display import Image, display

for fname, caption in [
    ('confusion_matrix_normalized.png', 'Confusion matrix (normalised)'),
    ('results.png',                     'Ultralytics summary'),
    ('PR_curve.png',                    'Precision-Recall curve'),
    ('F1_curve.png',                    'F1 vs confidence — pick your threshold here'),
    ('labels.jpg',                      'Label distribution & box sizes'),
]:
    p = RUN/fname
    if p.exists():
        print(f'\n=== {caption} ===')
        display(Image(filename=str(p), width=760))

# %%
# --- Cell 13: qualitative check + export ------------------------------------
import glob

for img in sorted(glob.glob(str(DS/'images'/'test'/'*')))[:6]:
    r = m.predict(img, imgsz=IMGSZ, conf=0.35, save=True, verbose=False)
    display(Image(filename=str(Path(r[0].save_dir)/Path(img).name), width=720))

# bundle weights + every chart for the report
import shutil as _sh
BUNDLE = Path('/content/senti_results'); BUNDLE.mkdir(exist_ok=True)
for f in ['training_curves.png', 'generalisation_gap.png', 'per_class.png',
          'confusion_matrix_normalized.png', 'results.png', 'PR_curve.png',
          'F1_curve.png', 'results.csv']:
    if (RUN/f).exists():
        _sh.copy2(RUN/f, BUNDLE/f)
_sh.copy2(best, BUNDLE/'best.pt')
_sh.make_archive('/content/senti_results', 'zip', BUNDLE)

from google.colab import files
files.download('/content/senti_results.zip')
