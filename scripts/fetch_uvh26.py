"""
fetch_uvh26.py
==============
Download the UVH-26 detector weights.

WHY THIS SCRIPT EXISTS
The weights are 57 MB of someone else's work, published by IISc AIM under
Apache 2.0. They are gitignored -- not ours to redistribute, and a repository
that carries other people's model checkpoints rots the moment they publish a
new version. But a config that points at a file no clone will ever have is
worse than useless, so the fetch has to live in the repo even though the file
does not.

WHY UVH-26 IS THE DEFAULT DETECTOR HERE
Measured on our own footage, same frame, same 0.35 confidence threshold:

    DriveIndia (our yolo11s fine-tune)   11 boxes
    UVH-26 (YOLOv11-S)                   52 boxes

and ours labelled a black SUV and a white sedan `bus` at 0.91 and 0.85. UVH-26
is trained on 26,646 frames from 2,800 Bengaluru Safe-City CCTV cameras, so an
elevated enforcement view is its native domain rather than a shift away from
the dashcam footage DriveIndia is built from.

WHAT IT COSTS
UVH-26 has 14 vehicle classes and NO pedestrian class. Any camera running
`triple_riding` -- the only rule that reads `result.pedestrians` -- must stay
on the DriveIndia weights. Every other rule works on either.

USAGE
    python scripts/fetch_uvh26.py
    python scripts/fetch_uvh26.py --model X        # the larger YOLOv11-X
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'models' / 'uvh26'
BASE = 'https://huggingface.co/iisc-aim/UVH-26/resolve/main'

FILES = {
    'S': ('weights/YOLOv11-S/UVH-26-MV-YOLOv11-S.pt', 57),
    'X': ('weights/YOLOv11-X/UVH-26-MV-YOLOv11-X.pt', 343),
}
EXTRA = ['uvh_classes.txt']


def get(rel: str, dest: Path, expect_mb: float | None = None) -> None:
    if dest.is_file() and dest.stat().st_size > 1000:
        print(f'  have  {dest.name} ({dest.stat().st_size/1e6:.0f} MB)')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    note = f' (~{expect_mb} MB)' if expect_mb else ''
    print(f'  fetch {dest.name}{note} ...', flush=True)
    urllib.request.urlretrieve(f'{BASE}/{rel}', dest)
    print(f'        done, {dest.stat().st_size/1e6:.0f} MB')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--model', choices=sorted(FILES), default='S',
                    help='S matches our yolo11s for a like-for-like comparison; '
                         'X is larger and slower')
    args = ap.parse_args()

    print(f'UVH-26 weights -> {DEST}')
    print('IISc AIM, Apache 2.0.  Dataset: CC BY 4.0, arXiv:2511.02563\n')

    rel, mb = FILES[args.model]
    get(rel, DEST / Path(rel).name, mb)
    for e in EXTRA:
        get(e, DEST / e)

    print('\nUsed by default in config/cam_demo.yaml, cam_junction.yaml,')
    print('cam_kolkata.yaml and cam_orr.yaml.')
    print('cam_davao.yaml stays on DriveIndia: triple_riding needs a pedestrian')
    print('class, which UVH-26 does not have.')


if __name__ == '__main__':
    main()
