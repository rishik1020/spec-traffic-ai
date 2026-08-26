# SPEC Traffic AI — Status

*Last updated: 30 July 2026*

An AI-powered traffic violation detection and enforcement platform for Indian
road conditions, with human-in-the-loop e-challan verification.

**Capstone Project-I, Woxsen University, School of Technology.**

---

## ✅ DONE

### 1. Detection model — TRAINED ✅
- **DriveIndia fine-tune complete**: `yolo11s`, 50 epochs, 7.7 h on RTX 4050 (6 GB)
- Dataset: 12,000 train / 1,500 val / 1,500 test, **21 classes** (pruned from 28 —
  7 classes had 0–22 boxes and would have scored 0.000, dragging the mean down)
- Official DriveIndia splits preserved (no re-splitting → no frame leakage)

| | epoch | mAP@50 | mAP@50-95 | precision | recall |
|---|---|---|---|---|---|
| **best** | 38 | **0.7157** | 0.5977 | 0.7210 | 0.6791 |
| final | 50 | 0.7004 | 0.5847 | 0.6714 | 0.6807 |

> ⚠️ Those are **validation** figures from `results.csv`. The **test** number —
> the one to quote against DriveIndia's published **78.7%** baseline — is the
> green bar in `results/driveindia/generalisation_gap.png`. Read it off before
> putting a number in the report or abstract.

- Charts: `results/driveindia/` — gap chart, per-class, training curves,
  confusion matrix, PR/F1 curves
- Earlier interim model (7 easy classes, Kaggle) scored 0.910 mAP@50 — **not
  comparable**; more classes and rarer ones lower the mean regardless of quality

### 2. Full pipeline built and verified end-to-end
```
video → detect → track → rule engine → rolling buffer → evidence package → (portal)
```

| File | Purpose |
|---|---|
| `senti/core/types.py` | `Detection` / `FrameResult` contract, class table, PCU weights |
| `senti/ingest/source.py` | file / RTSP / webcam behind one interface |
| `senti/perception/detector.py` | YOLO + supervision + HSV signal state |
| `senti/evidence/buffer.py` | rolling frame buffer |
| `senti/evidence/package.py` | evidence writer + **Evidence Defensibility Score** |
| `senti/rules/base.py` | `Rule` interface, registry, temporal consistency |
| `senti/rules/wrong_way.py` | reference rule |
| `senti/engine.py` | the wiring |
| `config/cam_demo.yaml` | per-camera profile |

**Verified output:**
```
WROTE : testcam_wrong_way_t7_f6
files : clip.mp4, evidence.json, frame_approach.jpg, frame_violation.jpg
reason: car (track #7) travelling 180 degrees against the permitted
        direction of flow, sustained over 72px of movement
EDS   : 66.6/100 → review | weakest=rule_margin
legal : MV Act s.184
```

### 3. Documents
- Abstract (191 words), PPT content, 8 verified research papers → `DATASETS.md`

---

## 🔨 TO DO — in order

### Immediate
1. ✅ done — training finished, charts generated
2. **Verify class names visually** — ids inferred from frequency ordering,
   NOT documented. Confirm `10=auto_rickshaw`, `9=traffic_light`, `0=pedestrian`.
   Ids 24–27 are undocumented in the paper entirely.
3. Retune `CLASS_CONF` in `detector.py` from the F1-vs-confidence curve

### ⚠️ BLOCKED ON YOU
4. **Record Hyderabad junction footage** — 2–5 min, phone from a footbridge is
   fine. **Get the signal head in frame** or red-light can't be demoed.
   Note which way traffic legally flows → that becomes `allowed_heading`.

### Next build (Phase 2–3)
5. **Calibration tool** — click-to-draw stop lines / lane polygons / direction
   arrows → per-camera YAML. *Everything geometric is blocked without this.*
6. `triple_riding` rule — `riders_on()` already exists, no calibration needed
7. `stop_line_crossing` — first user of supervision's `LineZone`
8. `red_light_jump` — `read_signal_state()` already exists
9. `over_speeding` — homography, **screening signal only, never legal evidence**
10. `no_helmet` — the ONLY rule needing another model (~1hr Colab, Roboflow)

### Phase 4
11. **Officer review portal** — queue → clip + reason trace + EDS → approve/reject
12. Challan generation (MV Act, VAHAN-style format)
13. ANPR — plate OCR with multi-frame voting (expect 70–85%, not 98%)

---

## 🎯 NOVELTY (the exam answer)

**Detection is NOT novel.** YOLO violation detection is heavily published;
DriveIndia publishes its own 78.7% baseline. Don't argue there.

**Real-world evidence (Kerala, 2023):** the Safe Kerala Project deployed **726
AI cameras** (KELTRON / Kerala MVD) detecting helmet, triple-riding, seatbelt,
phone-use, signal and speed violations. It detected **66.41 lakh violations** and issued **64.72 lakh challans** worth
**Rs 428.4 cr**, of which only **Rs 76.7 cr (18%)** was collected. **Detection was never
the bottleneck; enforceability was.** That is exactly the gap EDS addresses, and
it is the strongest available argument for both Industry Relevance and Innovation.

**The claim:**
> Existing systems detect violations. SPEC Traffic AI predicts whether the
> violation it detected can actually be *enforced*.

**Evidence Defensibility Score (EDS)** — the system scores its own evidence on
whether it would survive a legal challenge, *before* a human sees it:

| Dimension | Weight | Signal |
|---|---|---|
| plate | 0.30 | OCR agreement across frames, format validity |
| visibility | 0.20 | box area / occlusion at violation frame |
| track_integrity | 0.20 | frames tracked, ID switches |
| rule_margin | 0.15 | how far past threshold |
| context | 0.15 | HSV signal-state margin |

→ `≥80 auto` · `50–79 review` · `<50 drop`

**Detection confidence** says *"0.91 sure that's a motorcycle."*
**EDS** says *"this challan would be overturned — plate read from 3 frames with
2 disagreeing characters, 60% occluded at the violation moment."*
Nobody in this literature does the second.

**Validation experiment:** correlate EDS against real officer approve/reject
decisions (~200 packages), report ROC/AUC. If EDS predicts human rejection,
you've proven a machine can anticipate why enforcement evidence fails.

**Secondary novelty:** violations as a *control signal*, not just a fine —
e.g. "87% of red-light violations occur within 1.5s of phase change → extend
amber 3s→4s". PCU-weighted demand (IRC factors) is the India-correct way to
measure it; Western systems count vehicles, which treats 40 motorcycles and
40 buses identically.

---

## ⚠️ KNOWN CONSTRAINTS — be honest about these

| Constraint | Why |
|---|---|
| Vision speed is **screening only** | Legal speed needs radar/LIDAR. Point-to-point average speed IS defensible and camera-only. |
| **Cannot actuate real signals** | Safety-critical hardware. Deliverable = recommendation + SUMO simulation. |
| ANPR 70–85%, not 98% | Night glare, dust, damaged plates, two-wheeler occlusion. |
| 8 of 28 classes unusable | `ambulance`/`animal`/`police_vehicle` = 0 boxes. Report mAP over the ~14 viable classes alongside the all-class figure. |
| DriveIndia = academic use only | Fine for capstone; matters if productised. |
| `blur_number_plate` is a FEATURE | Don't merge with `number_plate` — it's the readable/unreadable triage signal the portal needs. |

---

## 🖥️ Environment

- RTX 4050 **6 GB** — batch 4 is the ceiling for `yolo11s@640`. **Close Overwolf /
  Adobe / browsers before training**; only ~750 MB headroom.
- Effective batch is already **64** via gradient accumulation (`nbs=64`), so a
  bigger `batch` would buy speed, not accuracy.
- torch 2.13.0+**cu126**, supervision 0.30.0, ultralytics 8.4.110

## Commands

```bash
# train
python scripts/train_local.py --data "C:\Users\Rishik Reddy\Downloads\driveindia_subset"

# charts only, no retrain
python scripts/train_local.py --data <dir> --eval-only

# run the pipeline (--device cpu keeps off a GPU that's training)
python run_senti.py --source data/videos/junction.mp4 --show
python run_senti.py --list-rules
```
