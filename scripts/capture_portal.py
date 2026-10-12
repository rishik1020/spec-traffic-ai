"""
capture_portal.py
=================
Screenshot a page of the running officer portal, for the report figures.

WHY NOT A PLAIN HEADLESS SCREENSHOT
The portal gates every page behind a reviewer identity held in an `officer`
cookie, set by a POST to /who. A one-shot `chrome --screenshot` cannot POST, so
it photographs the name gate instead of the page you asked for. This drives
Chrome over the DevTools Protocol, sets the cookie directly, and only then
navigates -- which keeps the gate in the product where it belongs rather than
weakening it for the sake of a figure.

USAGE
    python scripts/capture_portal.py --port 8077 \
        --path /review/kolkata_wrong_way_t15_f146__relaxed \
        --out docs/figures/fig5_portal.png

The portal must already be running on --port.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHROME_CANDIDATES = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
]


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    for n in ('chrome', 'chromium', 'msedge'):
        p = shutil.which(n)
        if p:
            return p
    raise SystemExit('no Chrome or Edge found')


class CDP:
    def __init__(self, ws_url: str):
        from websocket import create_connection
        # Chrome rejects a debug socket carrying an Origin header. Sending none
        # is preferable to starting the browser with --remote-allow-origins=*,
        # which would accept a connection from any page the browser loads.
        self.ws = create_connection(ws_url, timeout=30, suppress_origin=True)
        self.n = 0

    def call(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({'id': self.n, 'method': method,
                                 'params': params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get('id') == self.n:
                if 'error' in msg:
                    raise SystemExit(f'{method}: {msg["error"]}')
                return msg.get('result', {})

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8077, help='portal port')
    ap.add_argument('--path', default='/')
    ap.add_argument('--out', default='docs/figures/fig5_portal.png')
    ap.add_argument('--officer', default='Reviewer')
    ap.add_argument('--width', type=int, default=1440)
    ap.add_argument('--height', type=int, default=1120)
    ap.add_argument('--scale', type=float, default=2.0)
    ap.add_argument('--settle', type=float, default=3.0,
                    help='seconds to let media and layout settle')
    ap.add_argument('--full', action='store_true',
                    help='capture the full scrollable page')
    args = ap.parse_args()

    debug_port = 9222
    profile = ROOT / '.cache' / 'chrome-capture'
    profile.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [find_chrome(), '--headless=new', '--disable-gpu', '--no-sandbox',
         '--hide-scrollbars', '--no-first-run', '--disable-extensions',
         f'--remote-debugging-port={debug_port}',
         f'--user-data-dir={profile}', 'about:blank'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        ws_url = None
        for _ in range(60):
            try:
                raw = urllib.request.urlopen(
                    f'http://127.0.0.1:{debug_port}/json/list', timeout=2).read()
                tabs = json.loads(raw)
                pages = [t for t in tabs if t.get('type') == 'page']
                if pages:
                    ws_url = pages[0]['webSocketDebuggerUrl']
                    break
            except Exception:
                pass
            time.sleep(0.5)
        if not ws_url:
            raise SystemExit('Chrome did not expose a debugging target')

        cdp = CDP(ws_url)
        cdp.call('Page.enable')
        cdp.call('Network.enable')
        cdp.call('Emulation.setDeviceMetricsOverride',
                 width=args.width, height=args.height,
                 deviceScaleFactor=args.scale, mobile=False)

        # the identity the portal would otherwise collect at its name gate
        cdp.call('Network.setCookie', name='officer', value=args.officer,
                 domain='127.0.0.1', path='/')

        url = f'http://127.0.0.1:{args.port}{args.path}'
        cdp.call('Page.navigate', url=url)
        time.sleep(args.settle)

        shot = cdp.call('Page.captureScreenshot', format='png',
                        captureBeyondViewport=bool(args.full))
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(shot['data']))
        cdp.close()
        print(f'wrote {out}  ({out.stat().st_size / 1024:.0f} KB)  <- {url}')
    finally:
        proc.terminate()


if __name__ == '__main__':
    main()
