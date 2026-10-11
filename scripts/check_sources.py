"""
check_sources.py
================
Which of our candidate test sources can actually be used right now?

WHY A CHECKER AND NOT JUST A LIST
A list of traffic-camera URLs rots. Live streams end, channels go private,
aggregators rotate their feeds, and a camera that was perfect in daylight is
useless at 3am. A list that says "these work" is wrong within a week, and
discovering that an hour before a demo is the worst possible time.

So config/test_sources.yaml holds the candidates and this decides, today,
which are usable -- and for what. It answers the four questions that actually
determine whether a source is worth downloading:

  reachable     does it still resolve and open?
  daylight      is the scene bright enough for the detector to see anything?
  busy          how many objects per frame does our model actually find?
  signalised    is there a traffic light in frame, detected or drawable?

TIME ZONES ARE THE QUIET TRAP
A camera in California is in daylight while India sleeps. The one live
intersection we verified -- Fresno -- is bright roughly 21:30 to 05:30 IST, so
a demo at 11am IST shows a dark road. The report prints local time at the
source so that is visible before anyone builds a demo around it, rather than
after.

WHAT IT DOES NOT DO
It does not download anything. It probes a handful of frames and reports. Use
`yt-dlp --download-sections` to actually pull a clip once a source looks good.

USAGE
    python scripts/check_sources.py
    python scripts/check_sources.py --only live
    python scripts/check_sources.py --frames 6 --save-probes
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_SOURCES = ROOT / 'config' / 'test_sources.yaml'


def local_time(utc_offset_hours) -> str:
    if utc_offset_hours is None:
        return ''
    tz = timezone(timedelta(hours=float(utc_offset_hours)))
    now = datetime.now(tz)
    hour = now.hour
    light = 'daylight' if 7 <= hour < 18 else ('dusk' if 18 <= hour < 20 else 'NIGHT')
    return f'{now:%H:%M} local, {light}'


def brightness(frame) -> float:
    import cv2
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def probe(entry: dict, detector, n_frames: int, save_dir: Path | None) -> dict:
    """Open a source, sample a few frames, report what is actually in them."""
    import cv2
    from senti.ingest.source import resolve_source

    out = {'id': entry.get('id'), 'ok': False, 'note': ''}
    url = entry.get('url')
    if not url:
        out['note'] = 'no url'
        return out

    try:
        resolved = resolve_source(url)
    except Exception as e:
        out['note'] = f'resolve failed: {str(e)[:70]}'
        return out

    # same backend fix as VideoSource -- see senti/ingest/source.py
    cap = (cv2.VideoCapture(resolved, cv2.CAP_FFMPEG)
           if isinstance(resolved, str) and '://' in resolved
           else cv2.VideoCapture(resolved))
    if not cap.isOpened():
        out['note'] = 'could not open stream'
        cap.release()
        return out

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out.update(ok=True, width=w, height=h)

    objs, lights, bright, grabbed = [], 0, [], 0
    for i in range(n_frames * 25):
        ok, frame = cap.read()
        if not ok:
            break
        if i % 25:
            continue
        bright.append(brightness(frame))
        if detector is not None:
            r = detector.track(frame, frame_index=i, timestamp=i / 25.0)
            objs.append(len(r.detections))
            lights += len(r.traffic_lights)
        if save_dir is not None and grabbed == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_dir / f'{entry.get("id")}.jpg'), frame)
        grabbed += 1
        if grabbed >= n_frames:
            break
    cap.release()

    out['frames'] = grabbed
    if bright:
        out['brightness'] = round(sum(bright) / len(bright), 1)
    if objs:
        out['objects_per_frame'] = round(sum(objs) / len(objs), 1)
        out['traffic_lights'] = lights
    if grabbed == 0:
        out.update(ok=False, note='opened but no frames decoded')
    return out


def verdict(entry: dict, r: dict) -> str:
    if not r.get('ok'):
        return 'DEAD'
    bits = []
    if r.get('brightness', 255) < 55:
        bits.append('too dark now')
    if r.get('objects_per_frame', 0) < 1.5:
        bits.append('almost nothing detected')
    if entry.get('signalised') and not r.get('traffic_lights'):
        # Not fatal: the signal head can be drawn in calibration instead, which
        # is more reliable anyway. Worth flagging so nobody expects automatic
        # red-light detection to work out of the box.
        bits.append('signal not auto-detected (draw it with calibrate.py G)')
    return 'USABLE' if not bits else 'MARGINAL: ' + '; '.join(bits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--sources', default=str(DEFAULT_SOURCES))
    ap.add_argument('--only', default=None,
                    help='restrict to a kind: live, clip, dataset')
    ap.add_argument('--frames', type=int, default=4)
    ap.add_argument('--weights', default='runs/driveindia/weights/best.pt')
    ap.add_argument('--no-detect', action='store_true',
                    help='just check reachability, skip the model')
    ap.add_argument('--save-probes', action='store_true')
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.sources).read_text(encoding='utf-8')) or {}
    entries = [e for e in (cfg.get('sources') or [])
               if args.only in (None, e.get('kind'))]
    probeable = [e for e in entries if e.get('url') and e.get('kind') != 'dataset']

    print('=' * 78)
    print(f'CHECKING {len(probeable)} probeable source(s) of {len(entries)} listed')
    print('=' * 78)

    detector = None
    if not args.no_detect:
        from senti.perception.detector import TrafficDetector
        detector = TrafficDetector(weights=args.weights, imgsz=1280, device='cpu')

    save_dir = (ROOT / 'data' / 'probes') if args.save_probes else None
    results = []
    for e in probeable:
        lt = local_time(e.get('utc_offset'))
        print(f'\n{e.get("id")}  [{e.get("kind")}]  {e.get("place", "")}'
              + (f'   {lt}' if lt else ''))
        r = probe(e, detector, args.frames, save_dir)
        v = verdict(e, r)
        results.append((e, r, v))
        if r.get('ok'):
            print(f'  {r.get("width")}x{r.get("height")}  '
                  f'brightness {r.get("brightness", "?")}  '
                  f'objects/frame {r.get("objects_per_frame", "?")}  '
                  f'lights {r.get("traffic_lights", "?")}')
        print(f'  -> {v}' + (f'  ({r["note"]})' if r.get('note') else ''))

    print('\n' + '=' * 78)
    print('SUMMARY')
    print('=' * 78)
    for e, r, v in sorted(results, key=lambda x: (x[2].startswith('DEAD'),
                                                  x[2].startswith('MARGINAL'))):
        print(f'  {v.split(":")[0]:9s} {e.get("id"):28s} {e.get("place", "")}')

    datasets = [e for e in entries if e.get('kind') == 'dataset']
    if datasets:
        print('\nDATASETS (not probeable -- request or download by hand):')
        for d in datasets:
            print(f'  {d.get("id"):28s} {d.get("place", "")}')
            print(f'    {d.get("url", "")}')

    if save_dir:
        print(f'\nprobe frames -> {save_dir}')
    print('\nTo pull a clip from a source that looks good:')
    print('  yt-dlp -f "best[height<=720]" --download-sections "*0-180" \\')
    print('      -o "data/videos/NAME.mp4" "<url>"')


if __name__ == '__main__':
    main()
