# Senti Traffic — Datasets

Status of every dataset the model needs, and exactly what's blocking each one.

---

## ✅ On disk now

### DataCluster Indian number plates — **sample only**
- `datasets/plates/datacluster-sample/` — 47 images + 47 VOC annotations (147 MB)
- `datasets/plates/datacluster-sample/yolo/` — converted to YOLO, 37 train / 10 val, 1 class `number_plate`

> ⚠️ **This is a 47-image teaser, not the 16,192-image set the marketing quotes.**
> DataCluster is a commercial data vendor — they publish a small public sample and
> sell the full set. **Do not train on this.** It is useful only as a format
> reference and to smoke-test the training pipeline.

---

## 🔄 Interim substitutes while the DriveIndia EULA is pending

All need only a **free Kaggle token** — no EULA, no email wait.

> ⚠️ Specs below come from search-result summaries. Kaggle's pages are
> JS-rendered so they could not be verified directly. Check the real numbers
> on the dataset page before quoting any of this in the report.

### ⭐ 1. Traffic Vehicles Object Detection — best single substitute
- `saumyapatel/traffic-vehicles-object-detection`
- **7 classes: Car, Two Wheeler, Auto, Bus, Truck, Number Plate, Blur Number Plate**
- Reportedly **already in YOLOv5 format** — drops straight into the Colab notebook
- Why it's the best stand-in: it's the only set carrying **Auto (rickshaw)
  + Two Wheeler + Number Plate together**, so it closes the COCO auto-rickshaw
  gap *and* gives plate boxes in one training run.

### 2. Indian Driving Dataset (Kaggle mirror) — likely bypasses IDD registration
- `manjotpahwa/indian-driving-dataset`
- ~10,000 images, 34 classes, from the same 182 IDD drive sequences
- If genuine, this gets you IDD without the insaan.iiit.ac.in account
- ⚠️ Verify the licence on the Kaggle page — a mirror does not necessarily
  carry redistribution rights. Cite the original IIIT-H IDD paper regardless.

### 3. Indian Vehicle Dataset (DataCluster)
- `dataclusterlabs/indian-vehicle-dataset` — ~40k images, ~15k annotated

### 4. Roboflow top-ups (need `ROBOFLOW_API_KEY`)
- `autorickshawdetection/autorickshaw_detection` — 1,748 auto-rickshaw images
- `custom-yolo-orjlu/vehicle-detection-bnyxm` — ~1.1k imgs, 19 Indian classes

```bash
python scripts/fetch_datasets.py traffic-vehicles idd-kaggle
```

**Don't merge these with DriveIndia later.** Different class taxonomies and
labelling conventions — concatenating them produces silent label corruption.
Use them to prove the pipeline now; retrain cleanly on DriveIndia when it lands.

---

## 🔑 Needs your account key — script is ready

Run after adding credentials:

```bash
python scripts/fetch_datasets.py --all
```

| Dataset | Credential needed | Where to get it |
|---|---|---|
| Helmet / no-helmet (Roboflow ×2) | `ROBOFLOW_API_KEY` in `.env` | [app.roboflow.com](https://app.roboflow.com) → Settings → API keys |
| Indian plates (Kaggle) | `~/.kaggle/kaggle.json` | [kaggle.com](https://www.kaggle.com) → Settings → API → Create New Token |

Create a `.env` at the project root (it is gitignored):

```
ROBOFLOW_API_KEY=your_key_here
```

---

## ✋ Needs you in person — cannot be automated

These require reading and accepting a licence agreement. I won't create accounts,
sign EULAs, or submit forms on your behalf — download them yourself and drop them
in the folders below.

### 1. DriveIndia — ⭐ **the important one**
- **What:** 66,986 images · 1920×1080 · **24 classes incl. `Autorickshaw` and `Traffic light`** · already YOLO format
- **URL:** https://tihan.iith.ac.in/TiAND.html
- **Process:** download their EULA → sign it → upload it in the request form
- **Extract to:** `datasets/driveindia/`
- **Terms:** academic / non-commercial research only
- **Known gap:** night-time is underrepresented — matters for Indian enforcement

### 2. IDD — India Driving Dataset (IIIT Hyderabad)
- **What:** 182 drive sequences · 16 classes · unstructured Indian roads
- **URL:** http://idd.insaan.iiit.ac.in
- **Process:** register an account, accept the research licence
- **Extract to:** `datasets/idd/`
- **Note:** segmentation-first. Secondary to DriveIndia for detection work.

### 3. Autorickshaw Detection Challenge (CVIT)
- **What:** 1,000 images, 800 labelled, auto-rickshaws only
- **URL:** https://cvit.iiit.ac.in/autorickshaw_detection/
- **Process:** register on their Google Form; the link is emailed to you
- **Extract to:** `datasets/autorickshaw/`
- **Note:** only bother if DriveIndia's `Autorickshaw` class underperforms.

---

## 🚫 Deliberately not collected

### Triple riding — no usable public dataset exists
Every published paper used private footage (one used ~1.47M frames from Vellore
District Police; another hand-annotated an IDD subset). None is downloadable.

**You don't need one.** Detect `person` + `motorcycle` — both already available —
and count person-boxes associated to each motorcycle box. Geometry, not training.

### Traffic light *state* — wrong tool for the job
DriveIndia's `Traffic light` class gives you the **bounding box of the signal
head**, not which lamp is lit. Detection ≠ state.

Use the model to locate the signal head, then HSV colour-threshold the crop to
read red/amber/green. On an isolated ~30×80px box, classical CV beats a learned
classifier and is fully explainable — which matters when a challan is contested.

---

## Training order

1. **DriveIndia** → fine-tune `yolo11s` from COCO weights. Start with ~8–10k
   images, not all 67k at 1080p. This alone fixes the auto-rickshaw gap.
2. **Helmet** → separate small model, run only on motorcycle crops.
3. **Plates** → last, and only once the violation pipeline works end to end.
4. **Triple riding + signal state** → no training at all.

Two fine-tunes total. Nothing from scratch.
