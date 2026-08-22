"""
fetch_datasets.py
=================
Download the Senti Traffic training datasets that CAN be automated.

Three of the sources need an account key. Put them in a `.env` file at the
project root -- never commit it:

    ROBOFLOW_API_KEY=xxxxxxxxxxxx

and for Kaggle, place `kaggle.json` at  %USERPROFILE%\\.kaggle\\kaggle.json

Two sources CANNOT be automated at all -- they require you to read and accept a
licence agreement in person. Those are listed in DATASETS.md and this script
will simply tell you to go do them.

USAGE
    python scripts/fetch_datasets.py --list
    python scripts/fetch_datasets.py --all
    python scripts/fetch_datasets.py helmet plates
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"


# ---------------------------------------------------------------------------
# .env loading (no external dependency)
# ---------------------------------------------------------------------------

def load_env() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# Sources requiring a licence agreement -- HUMAN ONLY
# ---------------------------------------------------------------------------

MANUAL_SOURCES = {
    "driveindia": {
        "name": "DriveIndia (TiHAN, IIT Hyderabad)",
        "size": "66,986 images / 24 classes / YOLO format",
        "url": "https://tihan.iith.ac.in/TiAND.html",
        "why": "You must download the EULA, sign it, and upload it in their request form.",
        "dest": DATA / "driveindia",
    },
    "idd": {
        "name": "IDD - India Driving Dataset (IIIT Hyderabad)",
        "size": "182 drive sequences / 16 classes",
        "url": "http://idd.insaan.iiit.ac.in",
        "why": "Requires creating an account and agreeing to their research licence.",
        "dest": DATA / "idd",
    },
    "autorickshaw": {
        "name": "Autorickshaw Detection Challenge (CVIT)",
        "size": "1,000 images (800 labelled)",
        "url": "https://cvit.iiit.ac.in/autorickshaw_detection/",
        "why": "Access is granted by registering on a Google Form; link arrives by email.",
        "dest": DATA / "autorickshaw",
    },
}


# ---------------------------------------------------------------------------
# Automatable sources
# ---------------------------------------------------------------------------

ROBOFLOW_SOURCES = {
    "helmet": [
        # (workspace, project, version)
        ("gw-khadatkar-and-sv-wasule", "helmet-and-no-helmet-rider-detection", 1),
        ("spresearchwork", "motorcycle-riders-without-helmet", 1),
    ],
}

KAGGLE_SOURCES = {
    # ⭐ Closest interim substitute for DriveIndia. 7 classes, already YOLOv5
    #    format, and uniquely includes Auto + Two Wheeler + Number Plate in ONE
    #    dataset -- covers the auto-rickshaw gap AND gives plate boxes free.
    "traffic-vehicles": "saumyapatel/traffic-vehicles-object-detection",

    # IDD mirrored on Kaggle -- appears to bypass the insaan.iiit.ac.in
    # registration wall. ~10k images, 34 classes, from the 182 drive sequences.
    "idd-kaggle": "manjotpahwa/indian-driving-dataset",

    # DataCluster's full Indian vehicle set (~40k imgs, ~15k annotated).
    "vehicles-kaggle": "dataclusterlabs/indian-vehicle-dataset",

    # Plates -- the paid vendor's Kaggle listing.
    "plates-kaggle": "dataclusterlabs/indian-number-plates-dataset",
}


ROBOFLOW_EXTRA = {
    # Auto-rickshaw only, 1,748 images -- targeted top-up for the COCO gap.
    "autorickshaw": [("autorickshawdetection", "autorickshaw_detection", 2)],
    # General Indian road scenes with auto/bus/truck/car classes.
    "indian-roads": [("custom-yolo-orjlu", "vehicle-detection-bnyxm", 1)],
}


def fetch_roboflow(group: str) -> bool:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print(f"  ! ROBOFLOW_API_KEY not set -- skipping {group}")
        print("    get one free at https://app.roboflow.com  (Settings -> API keys)")
        return False

    try:
        from roboflow import Roboflow
    except ImportError:
        print("  ! pip install roboflow")
        return False

    dest = DATA / "helmet"
    dest.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=key)

    ok = True
    for workspace, project, version in ROBOFLOW_SOURCES[group]:
        print(f"  -> {workspace}/{project} v{version}")
        try:
            proj = rf.workspace(workspace).project(project)
            proj.version(version).download("yolov8", location=str(dest / project))
        except Exception as exc:
            print(f"    failed: {exc}")
            ok = False
    return ok


KAGGLE_DEST = {
    "traffic-vehicles": DATA / "interim" / "traffic-vehicles",
    "idd-kaggle":       DATA / "idd" / "kaggle",
    "vehicles-kaggle":  DATA / "interim" / "indian-vehicles",
    "plates-kaggle":    DATA / "plates" / "kaggle",
}


def fetch_kaggle(group: str) -> bool:
    slug = KAGGLE_SOURCES[group]
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        print(f"  ! {cred} not found -- skipping {group}")
        print("    kaggle.com -> Settings -> API -> Create New Token")
        return False

    dest = KAGGLE_DEST.get(group, DATA / "interim" / group)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  -> kaggle datasets download {slug}")
    try:
        subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download",
             "-d", slug, "-p", str(dest), "--unzip"],
            check=True,
        )
        return True
    except FileNotFoundError:
        print("  ! pip install kaggle")
    except subprocess.CalledProcessError as exc:
        print(f"  ! kaggle failed (exit {exc.returncode})")
    return False


def fetch_roboflow_extra(group: str) -> bool:
    """Roboflow sets that are top-ups rather than a primary training source."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print(f"  ! ROBOFLOW_API_KEY not set -- skipping {group}")
        return False
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  ! pip install roboflow")
        return False

    dest = DATA / "interim" / group
    dest.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=key)

    ok = True
    for workspace, project, version in ROBOFLOW_EXTRA[group]:
        print(f"  -> {workspace}/{project} v{version}")
        try:
            rf.workspace(workspace).project(project).version(version).download(
                "yolov8", location=str(dest / project)
            )
        except Exception as exc:
            print(f"    failed: {exc}")
            ok = False
    return ok


def fetch_hf_plate_sample() -> bool:
    """The public DataCluster sample -- ungated, no credentials needed.

    WARNING: this is a 47-image teaser, not the 16,192-image set advertised in
    their marketing. DataCluster is a commercial vendor; the full set is sold.
    Treat this as a format reference, NOT as training data.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ! pip install huggingface_hub")
        return False

    dest = DATA / "plates" / "datacluster-sample"
    print("  -> Dataclusterlabspvtltd/indian-number-plates-dataset (47-image sample)")
    snapshot_download(
        repo_id="Dataclusterlabspvtltd/indian-number-plates-dataset",
        repo_type="dataset",
        local_dir=str(dest),
    )
    return True


# ---------------------------------------------------------------------------

def show_manual() -> None:
    print("\n" + "=" * 72)
    print("REQUIRES YOU TO ACCEPT A LICENCE -- CANNOT BE AUTOMATED")
    print("=" * 72)
    for key, src in MANUAL_SOURCES.items():
        status = "PRESENT" if any(src["dest"].glob("*")) else "MISSING"
        print(f"\n[{status}] {src['name']}")
        print(f"    {src['size']}")
        print(f"    {src['url']}")
        print(f"    why manual: {src['why']}")
        print(f"    extract to: {src['dest']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="*", help="helmet | plates | plates-kaggle")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    load_env()

    if args.list or (not args.groups and not args.all):
        print("automatable groups:")
        print("  kaggle   : " + "  ".join(KAGGLE_SOURCES))
        print("  roboflow : helmet  " + "  ".join(ROBOFLOW_EXTRA))
        print("  no key   : plates (47-image HF sample only)")
        print("\nrecommended while DriveIndia is pending:")
        print("  python scripts/fetch_datasets.py traffic-vehicles idd-kaggle")
        show_manual()
        return

    groups = args.groups or ["traffic-vehicles", "idd-kaggle", "helmet", "plates"]

    for g in groups:
        print(f"\n[{g}]")
        if g == "helmet":
            fetch_roboflow("helmet")
        elif g == "plates":
            fetch_hf_plate_sample()
        elif g in KAGGLE_SOURCES:
            fetch_kaggle(g)
        elif g in ROBOFLOW_EXTRA:
            fetch_roboflow_extra(g)
        else:
            print(f"  ! unknown group: {g}")

    show_manual()


if __name__ == "__main__":
    main()
