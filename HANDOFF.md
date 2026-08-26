# SPEC Traffic AI — Handoff

*Written 26 August 2026. Read this first in a new session.*

Companion docs: [`STATUS.md`](STATUS.md) (roadmap) · [`PRESENTATION_NOTES.md`](PRESENTATION_NOTES.md) (Review-II talk track) · [`DATASETS.md`](DATASETS.md) (data sources) · [`README.md`](README.md) (public-facing)

**Repo:** https://github.com/rishik1020/spec-traffic-ai (public, MIT)
**Working dir:** `C:\Users\Rishik Reddy\Desktop\senti-traffic`
**Team:** Rishik Reddy P (23WU0102149), Rohan Reddy (23WU0102169), Hari Ramaneti (23WU0102135) · Mentor: Prof. Meher Gayatri

---

## 1. WHAT THIS PROJECT IS

Traffic violation detection for Indian roads, where the contribution is **not** detection
but **enforceability**. One-line claim:

> Existing systems detect violations. SPEC Traffic AI predicts whether the violation it
> detected can actually be *enforced*.

Naming history: `sentinel-traffic` → `sentinel-traffic-v2` → **abandoned**. Current build is
`senti-traffic` on disk, titled **SPEC Traffic AI** in all documents. Do not resurrect the
old folders.

---

## 2. CURRENT STATE

### Trained model ✅
`yolo11s` fine-tuned on DriveIndia. 50 epochs, 7.7 h, RTX 4050 (6 GB), batch 4.

| Split | mAP@50 | mAP@50-95 | precision | recall |
|---|---|---|---|---|
| train | 0.850 | 0.752 | 0.832 | 0.763 |
| val | 0.715 | 0.599 | 0.722 | 0.676 |
| **test** | **0.703** | 0.606 | 0.620 | 0.671 |

**0.703 is the reportable number.** Generalisation gap 0.147 (mild overfit).
Excluding 3 classes with <100 instances: **0.800** over 18 classes.
Over 7 enforcement classes: **0.868**.

Per-class highlights: `traffic_sign` .963 · `car` .961 · `motorcycle` .950 ·
**`auto_rickshaw` .888** · `pedestrian` .874 · `truck` .835 · `bus` .802 ·
**`traffic_light` .764**. Weakest: `construction_vehicle` .204, `tractor` .152,
`undocumented_26` .017 (all <100 boxes).

Weights: `runs/driveindia/weights/best.pt` (epoch 38) and `results/driveindia/best.pt`.
Charts in `results/driveindia/`. **`*.pt` is gitignored** — weights are NOT on GitHub.

### Pipeline ✅ built and verified end-to-end
```
video → detect → track → rule engine → rolling buffer → evidence package → (portal: NOT BUILT)
```

| File | Purpose |
|---|---|
| `senti/core/types.py` | `Detection`/`FrameResult` contract, class table, IRC PCU weights |
| `senti/ingest/source.py` | file · RTSP · webcam · YouTube behind one interface |
| `senti/perception/detector.py` | YOLO + supervision + HSV signal state |
| `senti/evidence/buffer.py` | rolling frame buffer |
| `senti/evidence/package.py` | evidence writer + **Evidence Defensibility Score** |
| `senti/rules/base.py` | `Rule` interface, registry, temporal consistency |
| `senti/rules/wrong_way.py` | the only implemented rule |
| `senti/calibration.py` | ⚠️ **built but NOT wired into any rule yet** |
| `senti/engine.py` | the wiring |

Scripts: `prepare_driveindia.py`, `prune_classes.py`, `train_local.py`, `train_colab.py`,
`voc_to_yolo.py`, `fetch_datasets.py`, `test_stream.py`, `triage_videos.py`, `calibrate.py`.

### Documents ✅
- Two-page capstone doc: `Downloads\SPEC Traffic AI - Capstone Review-II - Rishik Reddy P.docx`
  (183-word abstract, all four required elements, third person). **Team details still blank.**
- Deck: `Downloads\SPEC Traffic AI - Evaluation 2 FINAL.pptx` — 14 slides, 3 embedded charts,
  built from the user's Canva template by XML editing (styling preserved).

---

## 3. ⚠️ UNCOMMITTED WORK

```
 M senti/ingest/source.py          YouTube resolution via yt-dlp
?? scripts/test_stream.py          stream tester + subnet scan
?? scripts/triage_videos.py        rank videos by demo suitability
?? scripts/calibrate.py            interactive calibration tool
?? senti/calibration.py            Lane / StopLine / Homography model
?? results/driveindia/kolkata_annotated.jpg
```

**The user asked to hold back the live-stream files** (`source.py`, `test_stream.py`) while
still iterating. The calibration files are simply newer than the last commit. Confirm before
committing any of these.

---

## 4. THE CRITICAL OPEN ITEM

**`senti/calibration.py` is written but no rule uses it.** `wrong_way` still reads a single
global `allowed_heading` from config.

Why it matters: a live test on a two-way road produced **8 false positives** — vehicles
travelling legally were flagged, because one direction per camera cannot describe a road
where traffic moves both ways. **No value of that setting fixes it.**

The fix, roughly 20–30 min:
1. `Engine` builds `Calibration.from_dict(config['calibration'])`, passes it in `context`
2. `wrong_way.evaluate()` calls `cal.lane_at(det.bottom_center)`
3. If no lane → **return None (abstain)**, never a violation
4. Compare heading against **that lane's** heading, not a global one

`lane_at()` returning `None` outside every declared lane is deliberate: uncalibrated regions
must produce no verdict. Silence is cheap; a wrongful challan is not.

Once done, `stop_line_crossing` is nearly free — `StopLine.crossed(prev, curr)` is a sign
change of a 2D cross product, no thresholds needed.

---

## 5. KEY TECHNICAL FINDINGS

### ⭐ Viewpoint domain gap — the main finding
**DriveIndia is dashcam footage** (vehicle-mounted, ~1.5 m, forward-facing) captured for
autonomous-driving research. **Enforcement cameras look down from ~10 m.** From above, a car
roof and a bus roof are both large rectangles — which is why cars get labelled `bus` on
overhead footage. Verified by sampling training images directly.

**0.703 is valid within the dashcam domain and does not transfer to overhead CCTV.**
Not addressed in the reviewed literature. Present as a finding, never an apology.

### `imgsz` dominates on small-object footage
On the Kolkata clip: `imgsz=640` → 3 detections; `imgsz=1280` → **22**. The config default is
still 640. **Raise it to 1280 for any real footage.**

### Class difficulty beats class frequency
`marked_speed_bump` .823 from **124** boxes; `commercial_vehicle` .707 from **4,280**.
Visually distinctive classes learn from very few examples.

### Precision degrades faster than recall on unseen data
val .722 → test .620 (precision) vs .676 → .671 (recall). For enforcement this is the
dimension that matters — a missed violation harms nobody, a wrongful challan does.

### Class IDs were INFERRED, then verified
DriveIndia ships no class-name file. IDs were inferred from the paper's ordering cross-checked
against label frequencies, then **verified visually** by cropping real training boxes.
`car`, `bus`, `commercial_vehicle`, `truck`, `auto_rickshaw` all confirmed correct.
IDs 24–27 are undocumented in the paper entirely.

---

## 6. NOVELTY — Evidence Defensibility Score

**Real-world anchor (Kerala, 2023):** Safe Kerala Project deployed **726 AI cameras**;
detected **>1,00,000 violations**, issued **~3,000 challans**. A ~97% collapse.
**Detection was never the bottleneck; enforceability was.**

EDS scores every violation *before* an officer sees it:

| Dimension | Weight |
|---|---|
| plate certainty | 0.30 |
| subject visibility | 0.20 |
| track integrity | 0.20 |
| rule margin | 0.15 |
| context certainty | 0.15 |

→ `≥80` auto-fill · `50–79` review (names the weakest dimension) · `<50` drop

**Validation experiment (not yet run):** give ~200 evidence packages to a reviewer blind to
the score; measure whether EDS predicts approve/reject. Report AUC.

Secondary novelty: violations as a *control signal* (e.g. "87% of red-light violations occur
within 1.5 s of phase change → extend amber"), and PCU-weighted demand using IRC factors.
**Adaptive signal control alone is NOT novel** (SCATS/SCOOT/ATCS exist) — never pitch it as such.

---

## 7. HARD CONSTRAINTS — always state honestly

| Constraint | Detail |
|---|---|
| Vision speed = **screening only** | Legal speed needs radar/LIDAR. Point-to-point average speed IS camera-only and defensible. |
| **Cannot actuate signals** | Safety-critical hardware. Deliverable = recommendation + SUMO sim. |
| ANPR **70–85%** | Not the vendor-claimed 98%. |
| Software runs on a **box, not in the camera** | CCTV is a dumb sensor streaming RTSP. |
| DriveIndia = **academic use only** | Fine for capstone; matters if productised. |
| **ultralytics is AGPL-3.0** | Distributing a product triggers copyleft. Documented in LICENSE. |
| `blur_number_plate` is a **feature** | Don't merge with `number_plate` — it's the readable/unreadable triage signal. |

---

## 8. ENVIRONMENT & GOTCHAS

- **RTX 4050, 6 GB.** batch 4 is the ceiling for `yolo11s@640`. Close Overwolf / Adobe /
  Armoury Crate / browsers before training — headroom is ~750 MB.
- Effective batch is already **64** via gradient accumulation (`nbs=64`). Bigger `batch`
  buys speed, not accuracy.
- torch **2.13.0+cu126**, supervision 0.30.0, ultralytics 8.4.110, Python 3.10.11
- **`half` is removed in ultralytics 8.4** → use `quantize=16`. Already handled in
  `detector.py` with version detection.
- **`kaggle.json` is NOT installed locally** — the user used Colab Secrets. Kaggle API calls
  fail locally.
- **YouTube downloads are blocked** (403 on every player client). **Live streams still work**
  via HLS. `VideoSource` resolves YouTube URLs automatically.
- `pandas` is missing/ABI-broken — `train_local.py` deliberately uses stdlib `csv` instead.
- **gh auth flips to `prrmain` between shells.** Run `gh auth switch --user rishik1020`
  before any push, or you get a 403.
- Windows paths: quote them, and never pass `/c/Users/...` to OpenCV/Python (needs `C:\...`).

---

## 9. USER PREFERENCES (learned the hard way)

- **Stay inside the project directory.** Do not search Desktop/Downloads for test files —
  ask instead. Explicit paths the user supplies are fine.
- **Give commands to run, don't auto-run installs.** The user prefers executing them.
- **Never handle API keys or credentials.** They tried to paste a Kaggle key; declined.
- Explain the tech as work proceeds — the user is learning the stack, not just shipping.
- The user has felt overwhelmed before. Give clear structure and honest reassurance.

---

## 10. TEST ASSETS

- `data/videos/kolkata.mp4` — 10 s, 720×1280 portrait, **static camera**, 5 obj/frame at
  imgsz 1280, contains motorcycle + auto_rickshaw. **Best footage available.**
- Live YouTube traffic cams that worked: `sTF-6_xinUU`, `CXYr04BWvmc` (both 1280×720, 40+ fps)
- **Rejected datasets:** UniDataPro real-time-traffic (2 videos, cars only, CC-BY-NC-**ND**);
  FarzadNekouee density (single merged `Vehicle` class, aerial); DataCluster Kaggle sets
  (100-image teasers, not the advertised 16k/40k).

**Still needed:** real Hyderabad junction footage with a signal head in frame. Phone mounted
on a ledge is fine — mounted, not handheld, or all calibration assumptions break.

---

## 11. NEXT ACTIONS, IN ORDER

1. **Wire calibration into `wrong_way`** (see §4) — fixes the false positives
2. Set `imgsz: 1280` in `config/cam_demo.yaml`
3. Raise `'bus'` threshold in `detector.py` 0.45 → 0.65 (over-fires on large objects)
4. `triple_riding` rule — `riders_on()` already exists, needs no calibration
5. `stop_line_crossing` — uses `StopLine.crossed()`
6. `red_light_jump` — combines stop line + `read_signal_state()`
7. Helmet model (~1 hr Colab, Roboflow) → `no_helmet` rule
8. Officer review portal + challan generation
9. ANPR with multi-frame voting

---

## 12. VERIFIED CITATIONS (checked via arXiv/Crossref — not fabricated)

The three assigned papers:
1. Ravish, Rangaswamy & Char (2021), *Intelligent Traffic Violation Detection*, GCAT — `10.1109/GCAT52182.2021.9587520`
2. Ren, Y. (2024), IJCIS 17(40) — `10.1007/s44196-024-00427-6`
3. Kumar, Reddy & Rajalakshmi (2025), *DriveIndia* — `arXiv:2507.19912`

Supporting: Varma et al. IDD `arXiv:1811.10200` · Goyal et al. `arXiv:2204.08364` ·
Deshpande et al. `10.3389/frai.2025.1582257` · Cheng/Dong/Pang `10.1287/mnsc.2023.00575` ·
Liu/Gayah/Levin `arXiv:2406.19305`
