# SPEC Traffic AI

**Modular traffic violation detection for Indian road conditions, with evidence
defensibility scoring and human-in-the-loop e-challan verification.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Model](https://img.shields.io/badge/model-YOLO11-orange.svg)](https://github.com/ultralytics/ultralytics)

> B.Tech Capstone Project-I · School of Technology, Woxsen University

---

## The problem

India records the highest number of road-accident fatalities worldwide. Manual
enforcement cannot scale to Indian vehicle density, and the CCTV already
deployed is used for passive monitoring rather than enforcement.

Existing automated systems fit Indian roads badly:

- **COCO-trained detectors have no auto-rickshaw class** — structurally unable
  to enforce against a large share of Indian traffic
- Published systems are **monolithic** — one model, one violation, hardcoded geometry
- Most report **training-set accuracy**, not held-out performance
- They stop at *detection*, producing nothing that would survive being contested

## What makes this different

Detection is not the contribution — YOLO-based violation detection is heavily
published, and DriveIndia ships its own 78.7% baseline.

> **Existing systems detect violations.
> SPEC Traffic AI predicts whether the violation it detected can actually be _enforced_.**

### Evidence Defensibility Score (EDS)

Every violation is scored on whether it would survive a legal challenge —
*before* an officer ever sees it.

| Dimension | Weight | Signal |
|---|---|---|
| `plate` | 0.30 | OCR agreement across frames, format validity |
| `visibility` | 0.20 | subject box area / occlusion at the violation frame |
| `track_integrity` | 0.20 | frames tracked, ID switches in the evidence window |
| `rule_margin` | 0.15 | how far past the threshold the breach was |
| `context` | 0.15 | HSV signal-state margin, calibration freshness |

→ `≥80` auto-fill challan · `50–79` officer review · `<50` drop

Detection confidence says *"0.91 sure that's a motorcycle."*
EDS says *"this challan would be overturned — plate read from 3 frames with 2
disagreeing characters, 60% occluded at the moment of violation."*

Those come apart constantly, and only the second one decides whether a fine sticks.

---

## Architecture

```
video ─→ detect ─→ track ─→ rule engine ─→ rolling buffer ─→ evidence package ─→ portal
  │         │        │           │                │                  │
  │      YOLO11   ByteTrack   geometry +      last 5s held        clip + frames
  │    (the only  (Kalman +   temporal        in memory           + reason trace
  │     ML step)  Hungarian)  consistency                         + EDS
  │
  └─ file · RTSP · webcam — one interface, so live vs upload is a config string
```

**One shared perception pass, many small rules.** "Junction camera" vs "highway
camera" is a *YAML profile selecting which rules run* — not separate codebases.

**Third-party code is confined to the perception layer.** `supervision` is used
for detection parsing and filtering inside `senti/perception/`, then converted
to this project's own `Detection` objects. The rule engine imports no external
data model, which is what makes "swap the detector without touching the rules"
literally true rather than aspirational.

---

## Results

Baseline detector, held-out **test** split (not validation, not training):

| Metric | Score |
|---|---|
| mAP@50 | **0.910** |
| mAP@50-95 | 0.648 |
| Precision | 0.874 |
| Recall | 0.875 |
| train→test gap | **0.057** (healthy) |

Per-class highlights — `auto_rickshaw` is the strongest class, and the entire
reason an India-specific dataset was necessary:

| Class | mAP@50 |
|---|---|
| `auto` | **0.968** |
| `car` | 0.960 |
| `number_plate` | 0.934 |
| `two_wheeler` | 0.895 |
| `blur_number_plate` | 0.795 |

DriveIndia (21-class) training in progress.

---

## Quickstart

```bash
git clone https://github.com/rishik1020/spec-traffic-ai
cd spec-traffic-ai
pip install -r requirements.txt
```

GPU strongly recommended — install the CUDA build of torch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Run the pipeline:

```bash
python run_senti.py --list-rules
python run_senti.py --source data/videos/junction.mp4 --show
python run_senti.py --source rtsp://10.0.0.5:554/stream1 --config config/cam_hyd01.yaml
```

`--device cpu` forces CPU, useful when the GPU is busy training.

Each violation writes a self-contained folder:

```
data/evidence/demo01_wrong_way_t47_f3942/
├── clip.mp4              approach → violation → departure
├── frame_violation.jpg
├── frame_approach.jpg
└── evidence.json         reason trace, EDS breakdown, MV Act section
```

---

## Training

```bash
# 1. prepare a Colab-sized subset from the 11 GB DriveIndia release
python scripts/prepare_driveindia.py --zip path/to/DriveIndia.zip

# 2. drop classes with too few instances to learn (28 → 21)
python scripts/prune_classes.py --data path/to/driveindia_subset

# 3. train, then evaluate on the OFFICIAL test split
python scripts/train_local.py --data path/to/driveindia_subset_pruned --batch 4
```

Produces training curves, a **generalisation-gap chart** (same weights across
train/val/test), and a per-class breakdown with class support alongside accuracy.

Google Colab alternative: `scripts/train_colab.py` (cells marked with `# %%`).

---

## Adding a violation

One file, one YAML block. The registry is self-populating.

```python
# senti/rules/triple_riding.py
from ..core.types import Detection, FrameResult
from .base import Rule

class TripleRidingRule(Rule):
    name = 'triple_riding'
    applies_to = ('motorcycle',)
    stateful = False
    min_frames = 10
    mv_act_section = 'MV Act s.194C'
    description = 'More than two persons on a two-wheeler'

    def evaluate(self, det, result, context):
        riders = result.riders_on(det)
        if len(riders) < 3:
            return None
        return f'{len(riders)} riders on motorcycle #{det.track_id}', {
            'rider_count': len(riders),
        }
```

```yaml
# config/cam_hyd01.yaml
rules:
  triple_riding:
    min_frames: 10
```

Temporal consistency, cooldowns, evidence capture and EDS are inherited — a rule
only answers *"is this track violating right now?"*

---

## Project structure

```
senti/
├── core/types.py          Detection / FrameResult contract, IRC PCU weights
├── ingest/source.py       file · RTSP · webcam behind one interface
├── perception/detector.py YOLO + supervision + HSV signal state
├── evidence/buffer.py     rolling frame buffer
├── evidence/package.py    evidence writer + Evidence Defensibility Score
├── rules/base.py          Rule interface, registry, temporal consistency
├── rules/wrong_way.py     reference rule
└── engine.py              wiring
scripts/                   dataset prep, pruning, local + Colab training
config/                    per-camera profiles
```

---

## Violation coverage

| Violation | Extra model? | Calibration | Status |
|---|---|---|---|
| `wrong_way` | ❌ | direction vector | ✅ implemented |
| `triple_riding` | ❌ | none | 🔨 planned |
| `stop_line_crossing` | ❌ | stop-line polygon | 🔨 planned |
| `red_light_jump` | ❌ HSV reads the lamp | stop-line polygon | 🔨 planned |
| `over_speeding` | ❌ | homography | 🔨 planned |
| `no_helmet` | ⚠️ helmet classifier | none | 🔨 planned |
| seatbelt / phone use | — | — | ❌ camera-hardware problem |

Only **one** violation needs an additional model. The rest are geometry.

---

## Known constraints

Stated deliberately — a system that overclaims here is worse than useless.

- **Vision-based speed is a screening signal, never legal evidence.** Speed
  challans require radar/LIDAR. Point-to-point average speed *is* camera-only
  and defensible; instantaneous vision speed is not.
- **The system cannot actuate traffic signals.** Signal controllers are
  safety-critical hardware with conflict interlocks. Adaptive control is
  delivered as *recommendations* plus SUMO simulation.
- **ANPR reads 70–85% in real conditions**, not the 98% vendors claim — night
  glare, dust, damaged plates, two-wheeler occlusion.
- **`blur_number_plate` is deliberately a separate class.** It is not noise; it
  is the readable/unreadable triage signal the review portal depends on.
- **DriveIndia is licensed for academic, non-commercial research only.** It is
  not redistributed here.
- Class IDs for DriveIndia are **inferred** from the paper's ordering
  cross-checked against label frequencies — the release ships no class-name
  file, and IDs 24–27 are undocumented entirely.

---

## Dataset

Not included in this repository. Obtain it yourself:

| Dataset | Access |
|---|---|
| [DriveIndia](https://tihan.iith.ac.in/TiAND.html) (TiHAN, IIT Hyderabad) | EULA — 66,986 images, 24 classes |
| [IDD](http://idd.insaan.iiit.ac.in) (IIIT Hyderabad) | registration |

See [`DATASETS.md`](DATASETS.md) for the full list and access notes.

---

## Citation

```bibtex
@software{spec_traffic_ai_2026,
  author  = {Rishik Reddy P},
  title   = {SPEC Traffic AI: Evidence-Defensible Traffic Violation Detection
             for Indian Road Conditions},
  year    = {2026},
  url     = {https://github.com/rishik1020/spec-traffic-ai},
  note    = {B.Tech Capstone Project-I, Woxsen University}
}
```

Key references — see [`DATASETS.md`](DATASETS.md) for the full list:

- Kumar, Reddy & Rajalakshmi (2025). *DriveIndia*. [arXiv:2507.19912](https://arxiv.org/abs/2507.19912)
- Goyal et al. (2022). *Motorcycle Rider Traffic Violations on Unconstrained Roads*. [arXiv:2204.08364](https://arxiv.org/abs/2204.08364)
- Varma et al. (2019). *IDD*. [arXiv:1811.10200](https://arxiv.org/abs/1811.10200)
- Ravish, Rangaswamy & Char (2021). *Intelligent Traffic Violation Detection*. GCAT, IEEE

---

## License

This project's own code is released under the [MIT License](LICENSE).

> ⚠️ **Dependency licensing matters here.** Ultralytics YOLO is **AGPL-3.0**, a
> strong copyleft licence. Running this for research is fine, but *distributing*
> a product built on it triggers AGPL obligations — including releasing your
> source — unless you hold an Ultralytics Enterprise Licence. `supervision` is
> MIT and imposes no such condition. Anyone intending commercial use should
> resolve the Ultralytics licence first.

Datasets carry their own terms and are **not** covered by this licence.
