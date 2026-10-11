"""
record_when_ready.py
====================
Wait until a live camera is actually worth recording, then record it.

WHY THIS IS NOT JUST `yt-dlp`
Two things make live traffic cameras awkward to capture, and both were learned
the hard way today.

First, LIGHT. The best signalised junction available to this project is in
Fresno, twelve and a half hours behind IST, so it is dark through every Indian
working hour. Probed at 06:34 local the frame is pre-dawn: brightness 81, and
the detector found ZERO vehicles across three sampled frames. A recording made
then is 300 seconds of unusable footage that still takes 300 seconds to make.
So this polls until the scene passes a threshold the DETECTOR agrees with --
vehicles actually found, not just pixels bright enough.

Second, FRAGILITY. A 300-second capture attempted earlier today died at 40
seconds when the HLS segments began failing, and left an unfinalised .part file
with no moov atom that OpenCV refused to open at all. So this records several
SHORT clips rather than one long one, and remuxes each to browser-playable
H.264 as it goes. Losing one 120-second clip costs a clip; losing one
900-second clip costs the session.

USAGE
    python scripts/record_when_ready.py --url <live url> --name fresno
    python scripts/record_when_ready.py --url <url> --clips 3 --seconds 180
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def probe(url: str, detector, frames: int = 3) -> tuple[float, float]:
    """Mean brightness and mean vehicles per frame, right now."""
    import cv2

    from senti.ingest.source import resolve_source

    cap = cv2.VideoCapture(resolve_source(url), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return 0.0, 0.0

    bright, veh, got = [], [], 0
    for i in range(frames * 50):
        ok, f = cap.read()
        if not ok:
            break
        if i % 50:
            continue
        bright.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()))
        veh.append(len(detector.predict(f, imgsz=1280, conf=0.35,
                                        verbose=False)[0].boxes))
        got += 1
        if got >= frames:
            break
    cap.release()
    if not got:
        return 0.0, 0.0
    return sum(bright) / got, sum(veh) / got


def record(url: str, out: Path, seconds: int) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix('.part.mp4')
    cmd = ['yt-dlp', '--no-warnings', '-q',
           '-f', 'best[height<=720]/best',
           '--downloader', 'ffmpeg',
           '--downloader-args', f'ffmpeg_i:-t {seconds}',
           '-o', str(part), url]
    subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 180)

    # yt-dlp may leave the file under a slightly different name on failure
    src = part if part.is_file() else part.with_suffix('')
    if not src.is_file() or src.stat().st_size < 500_000:
        for leftover in part.parent.glob(part.stem + '*'):
            leftover.unlink(missing_ok=True)
        return False

    # Remux: an interrupted capture has no moov atom and will not open. This
    # also makes it H.264/faststart so the portal can play it.
    r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src),
                        '-c', 'copy', '-movflags', '+faststart', str(out)],
                       capture_output=True, text=True, timeout=300)
    src.unlink(missing_ok=True)
    return r.returncode == 0 and out.is_file() and out.stat().st_size > 100_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--url', required=True)
    ap.add_argument('--name', default='capture')
    ap.add_argument('--out-dir', default=str(ROOT / 'data' / 'videos'))
    ap.add_argument('--clips', type=int, default=3)
    ap.add_argument('--seconds', type=int, default=180)
    ap.add_argument('--min-vehicles', type=float, default=2.0,
                    help='mean vehicles per frame the DETECTOR must find')
    ap.add_argument('--min-brightness', type=float, default=95.0)
    ap.add_argument('--poll-minutes', type=float, default=10.0)
    ap.add_argument('--max-wait-hours', type=float, default=4.0)
    ap.add_argument('--weights', default='models/uvh26/UVH-26-MV-YOLOv11-S.pt')
    args = ap.parse_args()

    from ultralytics import YOLO
    det = YOLO(args.weights)
    try:
        det.to('cuda')
    except Exception:
        pass

    deadline = time.time() + args.max_wait_hours * 3600
    while True:
        b, v = probe(args.url, det)
        stamp = datetime.now().strftime('%H:%M')
        ready = v >= args.min_vehicles and b >= args.min_brightness
        print(f'[{stamp}] brightness {b:5.1f}  vehicles/frame {v:4.1f}  '
              + ('-> RECORDING' if ready else '-> waiting'), flush=True)
        if ready:
            break
        if time.time() > deadline:
            print('gave up waiting -- scene never got bright or busy enough')
            sys.exit(1)
        time.sleep(args.poll_minutes * 60)

    out_dir = Path(args.out_dir)
    made = []
    for i in range(1, args.clips + 1):
        dest = out_dir / f'{args.name}_{i}.mp4'
        print(f'  clip {i}/{args.clips} -> {dest.name} ({args.seconds}s) ...',
              flush=True)
        if record(args.url, dest, args.seconds):
            mb = dest.stat().st_size / 1e6
            print(f'    ok, {mb:.0f} MB', flush=True)
            made.append(dest)
        else:
            print('    FAILED (segments dropped) -- continuing', flush=True)

    print(f'\n{len(made)}/{args.clips} clip(s) recorded')
    for m in made:
        print('  ', m)
    if made:
        print('\nNext:')
        print(f'  python scripts/measure_headings.py --source {made[0]} '
              f'--config config/cam_junction.yaml --apply')
        print(f'  python scripts/calibrate.py --source {made[0]} '
              f'--config config/cam_junction.yaml      # G signal, S stop line, L lanes')


if __name__ == '__main__':
    main()
