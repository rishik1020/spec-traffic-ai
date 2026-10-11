"""
fetch_plates.py
===============
Download the licence-plate detector.

A challan is issued to a VEHICLE, so without a plate there is nobody to fine.
This model finds the plate; it does NOT read the characters, and that stopping
point is deliberate -- see senti/perception/plates.py.

morsetechlab/yolov11-license-plate-detection, AGPL-3.0, same licence as the
ultralytics runtime this project already depends on.

USAGE
    python scripts/fetch_plates.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'models' / 'plates' / 'license-plate-v1s.pt'
URL = ('https://huggingface.co/morsetechlab/yolov11-license-plate-detection'
       '/resolve/main/license-plate-finetune-v1s.pt')

if DEST.is_file() and DEST.stat().st_size > 1000:
    print(f'have  {DEST.name} ({DEST.stat().st_size/1e6:.0f} MB)')
else:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f'fetch {DEST.name} (~19 MB) ...', flush=True)
    urllib.request.urlretrieve(URL, DEST)
    print(f'      done, {DEST.stat().st_size/1e6:.0f} MB')
print('\nPlate detection is ON by default. Disable per camera with:\n  plates: false')
