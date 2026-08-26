"""
test_stream.py
==============
Verify a live camera stream before pointing the pipeline at it.

WHY A SEPARATE TOOL
When `run_senti.py` fails on a live source there are three possible causes --
the network, the stream format, or the pipeline. This isolates the first two, so
a failure here means "the camera is not reachable" and a success means "any
remaining problem is ours."

It also measures ACHIEVED fps, which matters: a camera advertising 30fps that
actually delivers 8 over Wi-Fi will silently starve the tracker, and tracking
gaps look exactly like detection failures.

USAGE
    python scripts/test_stream.py --url http://192.168.1.50:8080/video
    python scripts/test_stream.py --url rtsp://192.168.1.50:554/h264 --preview
    python scripts/test_stream.py --scan            # find cameras on this subnet
"""

from __future__ import annotations

import argparse
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

# Ports commonly used by phone IP-camera apps and real CCTV.
CAMERA_PORTS = [
    (8080, 'http',  '/video'),        # IP Webcam (Android) -- most common
    (4747, 'http',  '/video'),        # DroidCam
    (8554, 'rtsp',  '/live'),         # RTSP server apps
    (554,  'rtsp',  '/stream1'),      # real CCTV / NVR default
    (80,   'http',  '/video'),
]


def local_subnet() -> str:
    """Best-guess /24 this machine sits on."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))          # no packets sent; just picks the iface
        ip = s.getsockname()[0]
    finally:
        s.close()
    return '.'.join(ip.split('.')[:3])


def port_open(host: str, port: int, timeout: float = 0.35) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def scan(subnet: str) -> None:
    print(f'scanning {subnet}.1-254 for camera ports {[p for p, _, _ in CAMERA_PORTS]} ...\n')
    found = []

    def probe(n: int):
        host = f'{subnet}.{n}'
        for port, scheme, path in CAMERA_PORTS:
            if port_open(host, port):
                found.append((host, port, scheme, path))

    with ThreadPoolExecutor(max_workers=128) as pool:
        list(pool.map(probe, range(1, 255)))

    if not found:
        print('  nothing found.')
        print('  -> is the phone on the SAME Wi-Fi (not mobile data)?')
        print('  -> is the camera app actually streaming?')
        print('  -> some routers block device-to-device traffic (AP isolation);')
        print('     check for a "client isolation" setting.')
        return

    print('  candidates:')
    for host, port, scheme, path in sorted(found):
        print(f'    {scheme}://{host}:{port}{path}')
    print('\n  test one with:  python scripts/test_stream.py --url <url>')


def probe_stream(url: str, seconds: float, preview: bool) -> bool:
    # accept a YouTube watch URL directly
    import sys as _s, pathlib as _p
    _s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
    from senti.ingest.source import resolve_source
    url = resolve_source(url)
    print(f'opening {url[:90]}{"..." if len(url) > 90 else ""}')
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(url)
    open_ms = (time.perf_counter() - t0) * 1000

    if not cap.isOpened():
        print('  FAILED to open.\n')
        print('  checklist:')
        print('    - phone and laptop on the same Wi-Fi?')
        print('    - URL exactly as the app displays it (path matters: /video, /h264, ...)')
        print('    - try the URL in a browser first -- if the browser cannot play it,')
        print('      OpenCV will not either')
        print('    - Windows Firewall may block inbound; allow python.exe on private networks')
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f'  connected in {open_ms:.0f} ms')
    print(f'  resolution : {w}x{h}')
    print(f'  declared   : {declared:.1f} fps' if declared > 0
          else '  declared   : unknown (normal for RTSP)')

    print(f'\nreading for {seconds:.0f}s ...')
    frames, dropped = 0, 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            dropped += 1
            if dropped > 30:
                print('  stream died mid-read.')
                break
            continue
        frames += 1
        if preview:
            cv2.imshow('stream test  (q to quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    elapsed = time.perf_counter() - start
    cap.release()
    if preview:
        cv2.destroyAllWindows()

    achieved = frames / elapsed if elapsed else 0
    print(f'  frames     : {frames} in {elapsed:.1f}s')
    print(f'  ACHIEVED   : {achieved:.1f} fps')
    print(f'  read fails : {dropped}')

    print()
    if achieved >= 20:
        print('  VERDICT: good -- full pipeline should keep up.')
    elif achieved >= 10:
        print('  VERDICT: usable, but tracking will be less stable.')
        print('           Lower the app resolution, or set stride: 1 and accept it.')
    elif achieved > 0:
        print('  VERDICT: too slow. Drop the app to 640x480, move closer to the')
        print('           router, or record to file and process offline instead.')
    else:
        print('  VERDICT: opened but delivered nothing -- wrong path in the URL?')

    if w and h and w * h > 1920 * 1080:
        print(f'  NOTE: {w}x{h} is heavy. 1280x720 is plenty for detection and')
        print('        will stream far more reliably.')
    return frames > 0


def main() -> None:
    ap = argparse.ArgumentParser(description='Test a live camera stream')
    ap.add_argument('--url', help='http://... or rtsp://...')
    ap.add_argument('--scan', action='store_true', help='find cameras on this subnet')
    ap.add_argument('--seconds', type=float, default=8.0)
    ap.add_argument('--preview', action='store_true', help='show the video window')
    args = ap.parse_args()

    if args.scan:
        scan(local_subnet())
        return
    if not args.url:
        ap.error('give --url, or --scan to look for one')

    ok = probe_stream(args.url, args.seconds, args.preview)
    if ok:
        print('\nnext:')
        print(f'  python run_senti.py --source "{args.url}" --show')


if __name__ == '__main__':
    main()
