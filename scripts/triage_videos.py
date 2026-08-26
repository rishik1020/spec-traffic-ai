"""
triage_videos.py
================
Scan a folder of videos and rank them by how usable they are for this project.

WHY
A 500-video dataset is far too many to watch. Most clips will be useless here --
wrong viewpoint, no two-wheelers, camera panning, too dark. This samples a few
frames from each, runs the real detector over them, and scores each clip on the
things that actually decide whether a demo works:

  static camera   -- geometric rules (stop line, lane polygons, homography) are
                     MEANINGLESS on a moving camera. This is the hard gate.
  traffic_light   -- without one in frame, red_light_jump cannot be demonstrated
  motorcycle      -- gates triple_riding and (later) no_helmet
  auto_rickshaw   -- the Indian class that justifies the whole dataset choice
  object density  -- a busy scene shows more in a 30-second demo
  resolution      -- small objects need pixels

USAGE
    python scripts/triage_videos.py --dir data/videos
    python scripts/triage_videos.py --dir <dir> --samples 8 --top 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def camera_is_static(cap, n_pairs: int = 4) -> tuple[bool, float]:
    """Estimate whether the camera itself moves.

    Compares consecutive frames far apart in the clip. On a fixed camera the
    background is identical and only vehicles differ, so the median absolute
    difference stays low. On a dashcam or a pan, EVERYTHING shifts and the
    median jumps. Median, not mean, so a few large moving vehicles do not
    swamp a genuinely static scene.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 20:
        return False, 999.0

    diffs = []
    for i in range(n_pairs):
        pos = int(total * (i + 1) / (n_pairs + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok1, f1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos + 3)
        ok2, f2 = cap.read()
        if not (ok1 and ok2):
            continue
        g1 = cv2.cvtColor(cv2.resize(f1, (320, 180)), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(f2, (320, 180)), cv2.COLOR_BGR2GRAY)
        diffs.append(float(np.median(cv2.absdiff(g1, g2))))

    if not diffs:
        return False, 999.0
    med = float(np.median(diffs))
    return med < 4.0, med          # 4/255 -- empirical, generous


def score(stats: dict) -> float:
    """Weighted suitability, 0-100. Static camera dominates on purpose."""
    s = 0.0
    s += 40 if stats['static'] else 0          # hard requirement for geometry
    s += 15 if stats['has_traffic_light'] else 0
    s += 15 if stats['has_motorcycle'] else 0
    s += 10 if stats['has_auto'] else 0
    s += min(10, stats['avg_objects'])          # density, capped
    s += 10 if stats['height'] >= 720 else (5 if stats['height'] >= 480 else 0)
    return round(s, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='folder of videos')
    ap.add_argument('--weights', default='runs/driveindia/weights/best.pt')
    ap.add_argument('--samples', type=int, default=5, help='frames per video')
    ap.add_argument('--imgsz', type=int, default=1280,
                    help='1280 finds far more small objects than 640')
    ap.add_argument('--top', type=int, default=12)
    ap.add_argument('--limit', type=int, default=None, help='cap videos scanned')
    args = ap.parse_args()

    from senti.perception.detector import TrafficDetector

    root = Path(args.dir)
    vids = sorted(p for p in root.rglob('*') if p.suffix.lower() in VIDEO_EXT)
    if args.limit:
        vids = vids[:args.limit]
    if not vids:
        raise SystemExit(f'no videos under {root}')

    print(f'{len(vids)} videos, sampling {args.samples} frames each @ imgsz={args.imgsz}\n')
    det = TrafficDetector(weights=args.weights, imgsz=args.imgsz)

    results = []
    for n, v in enumerate(vids, 1):
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        static, motion = camera_is_static(cap)

        counts, objs = {}, []
        for i in range(args.samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 1) / (args.samples + 1)))
            ok, frame = cap.read()
            if not ok:
                continue
            r = det.detect(frame)
            objs.append(len(r.detections))
            for d in r.detections:
                counts[d.cls_name] = counts.get(d.cls_name, 0) + 1
        cap.release()

        st = {
            'path': v, 'width': w, 'height': h,
            'seconds': total / fps if fps else 0,
            'static': static, 'motion': motion,
            'avg_objects': (sum(objs) / len(objs)) if objs else 0,
            'has_traffic_light': counts.get('traffic_light', 0) > 0,
            'has_motorcycle': counts.get('motorcycle', 0) > 0,
            'has_auto': counts.get('auto_rickshaw', 0) > 0,
            'counts': counts,
        }
        st['score'] = score(st)
        results.append(st)

        if n % 25 == 0 or n == len(vids):
            print(f'  scanned {n}/{len(vids)}')

    results.sort(key=lambda r: -r['score'])

    print(f'\n{"="*100}\nTOP {args.top} CANDIDATES\n{"="*100}')
    print(f'{"score":>5} {"static":>7} {"res":>10} {"secs":>6} {"obj/f":>6}  '
          f'{"light":>5} {"moto":>5} {"auto":>5}  file')
    for r in results[:args.top]:
        print(f'{r["score"]:5.1f} {"YES" if r["static"] else "no":>7} '
              f'{r["width"]}x{r["height"]:<5} {r["seconds"]:6.0f} '
              f'{r["avg_objects"]:6.1f}  '
              f'{"Y" if r["has_traffic_light"] else "-":>5} '
              f'{"Y" if r["has_motorcycle"] else "-":>5} '
              f'{"Y" if r["has_auto"] else "-":>5}  {r["path"].name[:44]}')

    n_static = sum(1 for r in results if r['static'])
    n_light = sum(1 for r in results if r['has_traffic_light'])
    n_moto = sum(1 for r in results if r['has_motorcycle'])
    print(f'\nof {len(results)} videos: {n_static} static camera | '
          f'{n_light} with traffic light | {n_moto} with motorcycles')

    if results:
        best = results[0]
        print(f'\nbest candidate: {best["path"]}')
        print(f'  classes seen: {dict(sorted(best["counts"].items(), key=lambda x: -x[1]))}')
        print(f'\n  python run_senti.py --source "{best["path"]}" --show')
        if not best['static']:
            print('\n  *** WARNING: even the best clip has a MOVING camera. Geometric')
            print('      rules (stop line, lanes, speed) cannot be calibrated on it.')


if __name__ == '__main__':
    main()
