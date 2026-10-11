# Final Report — spec, and what we already have for each chapter

*From `Capstone Project guidelines`, Woxsen School of Technology.*
*Written 11 Oct 2026. Report not started; this is the outline to write against.*

---

## Hard requirements

| | |
|---|---|
| **Word count** | **3,000–4,000** excluding references, appendices, diagrams |
| **Similarity** | **<15%** on Turnitin. Non-compliance = disqualification |
| **References** | **15–20 minimum**, IEEE format, preferably last 5–10 years |
| **Font** | Times New Roman, 12 pt body |
| **Spacing** | 1.5 throughout |
| **Margins** | 1 inch all sides |
| **Alignment** | Justified |
| **Page numbers** | Every page, from the Introduction |
| **Headings** | H1 bold 14 pt numbered · H2 bold 13 pt · H3 bold 12 pt |
| **Figures** | ≥300 DPI, caption **below**, 10 pt, numbered, referenced in text before they appear, centred |
| **Tables** | Caption **above**, 10 pt |
| **Voice** | **Third person. No "I", "we", "our".** Formal, active where possible |

### Chapter word budget

| Chapter | Share | Words (of 3,500) |
|---|---|---|
| 1 Introduction | 10–12% | 350–420 |
| 2 Technology Review | 20–25% | 700–875 |
| 3 Objectives & Methodology | 20–25% | 700–875 |
| 4 Implementation & Results | 25–30% | 875–1,050 |
| 5 Conclusion & Future Work | 10–12% | 350–420 |

### Marks

| Criterion | % | What it rewards |
|---|---|---|
| Problem Identification | 20 | Clarity, real-world significance, scope |
| **Methodology** | **25** | Technical approach, tool selection, systematic process |
| **Innovation & Industry Relevance** | **25** | Novelty, gap addressing, practical impact |
| Presentation Clarity | 15 | Organisation, visuals, delivery, Q&A |
| Report Quality | 15 | Formatting, writing, completeness, accuracy |

Methodology and Innovation are half the marks. This project's strongest
material is measured findings and design decisions with reasons — which is
exactly what those two criteria reward, so the report should lead with them
rather than with feature lists.

---

## What exists for each chapter

### 1. Introduction
- India has the highest road-accident fatalities worldwide
- MoRTH 2023: overspeeding in **68.4%** of accidents; **54,568** killed without a helmet
- Existing CCTV is passive; manual enforcement cannot scale
- COCO-trained detectors have **no auto-rickshaw class** — structurally unable to
  enforce against a large share of Indian traffic

### 2. Technology Review *(the deck's table is the spine)*
- DriveIndia (arXiv:2507.19912) — 78.7% mAP@50, dashcam
- Ravish et al. GCAT 2021 — end-to-end pipeline, no held-out evaluation
- Ren 2024 IJCIS — >96.9% accuracy, **no held-out split, no field deployment**
- BMD-45 (arXiv:2604.24419) — 83.8% in-domain vs 33.6% cross-domain
- UVH-26 (arXiv:2511.02563) — 26,646 frames, 2,800 Bengaluru Safe-City cameras
- Wilson et al. Cochrane 2010 — 35 studies, crashes down 8–49%
- Høye 2015 — spot cameras −51% fatal, section control −56% KSI
- Rhead et al. 2012 — ANPR 28.5% misread *(UK 2012; present as a floor)*
- Webster 1958 · IRC:93 · MV Act · s.65B / BSA 2023 s.63

**Reference gap CLOSED — verified on arXiv 11 Oct 2026, not guessed:**

- Zhang, Sun, Jiang, Yu et al., *ByteTrack: Multi-Object Tracking by
  Associating Every Detection Box*, **arXiv:2110.06864** (2021).
  Cite where the report states tracking is a Kalman filter and Hungarian
  matching rather than a learned model.
- Khanam & Hussain, *YOLOv11: An Overview of the Key Architectural
  Enhancements*, **arXiv:2410.17725** (2024).
  Cite for the detector architecture, which both our fine-tune and UVH-26 share.
- Sochor, Juránek & Herout, *Traffic Surveillance Camera Calibration by 3D
  Model Bounding Box Alignment*, **arXiv:1702.06451** (2017).
- Jain & Saunier, *Autocamera Calibration for Traffic Surveillance Cameras
  with Wide Angle Lenses*, **arXiv:2001.07243** (2020).
  Both belong in Future Work: automatic calibration is the step that turns
  "works on a camera" into "works on cameras", and `measure_headings.py` is a
  first move in that direction. Citing prior art here shows the gap was
  surveyed rather than assumed.

That is **17**, clearing the 15 minimum with room to drop one.

### 3. Objectives & Methodology
- One shared perception pass, many independent consumers
- Rule interface + **temporal consistency in the base class** — the single
  biggest false-positive killer
- **Abstention as a design principle**: `lane_at` → None, `speed_kmph` → None,
  warming-up → keep the existing plan. Silence is cheap; a wrongful challan is not
- Per-lane calibration; headings **measured, not drawn** (`measure_headings.py`)
- Signal state by HSV threshold over a **drawn** head, not a classifier —
  explainable when contested
- EDS: five weighted dimensions, renormalised over what was measurable
- Webster for signal timing — arithmetic an engineer can check by hand

### 4. Implementation & Results *(strongest chapter — all measured)*

| Finding | Number |
|---|---|
| Detector, held-out test | **0.703** mAP@50 all 21 classes; 0.868 enforcement classes; gap 0.147 |
| Viewpoint domain gap | ours **11** boxes vs UVH-26 **52** on the same frame; two cars labelled `bus` at 0.91/0.85 |
| False positives from a guessed heading | **15 → 1** after measuring (coherence 0.30, 35% opposing) |
| Rider detection | **1,467** motorcycles across two clips, **never** two people on one bike |
| Plate availability | **3 of 21** violations had a plate; widths 17/23/41 px, **all unreadable** |
| Real enforcement cameras | 33 plates / 40 Bengaluru frames; median **36 px**; only **30%** ≥45 px |
| Latency | **18.0 ms/frame** (55.6 fps) RTX 4050 @1280 vs 304 ms CPU |
| Rule logic | **22/22** assertions, no footage required |
| Signal controller | 4 scenarios; green follows queue 37→57 s; empty arm still served |

Figures available: `results/driveindia/generalisation_gap.png`, `per_class.png`,
`confusion_matrix_normalized.png`, `training_curves.png`, `signal_split.png`,
`kolkata_annotated.jpg`, plus portal screenshots.

**GAP: EDS validation (AUC) not run — blocked on footage.** It is the single
result that would most strengthen Innovation.

### 5. Conclusion & Future Work
- ANPR with multi-frame voting; a **dedicated plate camera per lane**, since the
  camera that sees the offence and the one that identifies the vehicle are
  different instruments
- Automatic calibration — infer lane polygons from accumulated flow;
  `measure_headings.py` is the first step
- Rider-annotated training data to recover triple riding and helmet compliance
- EDS validated against serving officers rather than students

---

## Before writing

1. **Third person throughout.** Most of our notes say "we" — that has to go.
2. **3,000–4,000 words is tight.** The measured findings earn the space; feature
   descriptions do not.
3. **Every figure referenced in the text before it appears**, caption below, 10 pt.
4. **Turnitin <15%** — do not paste from the deck or from HANDOFF.md.
5. **Honest limits stated**: student reviewers, one camera, one rule, small sample.
   Stating them is a Methodology mark, not a concession.
