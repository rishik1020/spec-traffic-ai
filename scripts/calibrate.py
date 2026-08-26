"""
calibrate.py
============
Draw a camera's geometry on a real frame, and write it into the YAML.

    python scripts/calibrate.py --source data/videos/kolkata.mp4 --config config/cam_demo.yaml

CONTROLS
    L      lane mode      -- click 3+ corners of a lane, then D to close it.
                             You are then asked to click ONE arrow point showing
                             the direction traffic is permitted to travel, and
                             to type which ARM of the junction the lane feeds
                             (north/south/east/west). The arm is what adaptive
                             signal timing measures congestion against; leave it
                             blank on a highway, where there is nothing to time.
    S      stop-line mode -- click 2 points.
    H      homography     -- click 4 road-surface points, then type their real
                             world metre coordinates in the terminal.
    D      done with the current shape
    U      undo last point (or last completed shape if none pending)
    C      clear everything
    [ / ]  step backward / forward through the video to find a good frame
    W      write to the config YAML
    Q      quit without saving

WHY A FRAME AND NOT A GUESS
Every geometric rule is defined in image coordinates, which depend entirely on
where this camera sits. Typing polygon coordinates by hand is guesswork; drawing
them on the actual frame takes a minute and is exact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODE_LANE, MODE_STOP, MODE_HOM = 'lane', 'stop', 'homography'
COL = {MODE_LANE: (0, 220, 0), MODE_STOP: (0, 165, 255), MODE_HOM: (255, 120, 0)}


class Calibrator:
    def __init__(self, frame):
        self.base = frame
        self.h, self.w = frame.shape[:2]
        self.mode = MODE_LANE
        self.pending: list[tuple[int, int]] = []
        self.lanes: list[dict] = []
        self.stops: list[dict] = []
        self.hom_src: list[tuple[int, int]] = []
        self.hom_dst: list[tuple[float, float]] = []
        self.awaiting_heading: dict | None = None
        self.msg = ''

    # -- mouse -------------------------------------------------------------

    def on_mouse(self, event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.awaiting_heading is not None:
            lane = self.awaiting_heading
            cx = sum(p[0] for p in lane['polygon']) / len(lane['polygon'])
            cy = sum(p[1] for p in lane['polygon']) / len(lane['polygon'])
            dx, dy = x - cx, y - cy
            mag = (dx * dx + dy * dy) ** .5 or 1
            lane['heading'] = [round(dx / mag, 3), round(dy / mag, 3)]
            # Which ARM of the junction this lane feeds. Adaptive signal timing
            # measures congestion per arm, so a lane with no arm contributes to
            # nobody's demand. Blank is correct on a highway -- there is no
            # junction to time.
            try:
                arm = input("  which approach/arm is this lane? "
                            "(north/south/east/west, blank = none): ").strip()
            except EOFError:
                arm = ''
            if arm:
                lane['approach'] = arm
            self.lanes.append(lane)
            self.awaiting_heading = None
            self.msg = f"lane '{lane['name']}' heading {lane['heading']}"
            return

        self.pending.append((x, y))
        if self.mode == MODE_HOM and len(self.pending) == 4:
            self._finish()

    # -- shape completion --------------------------------------------------

    def _finish(self):
        if self.mode == MODE_LANE:
            if len(self.pending) < 3:
                self.msg = 'a lane needs at least 3 points'
                return
            lane = {'name': f'lane{len(self.lanes)+1}', 'polygon': list(self.pending)}
            self.pending = []
            self.awaiting_heading = lane
            self.msg = 'now click ONE point in the direction traffic may travel'

        elif self.mode == MODE_STOP:
            if len(self.pending) != 2:
                self.msg = 'a stop line needs exactly 2 points'
                return
            self.stops.append({'name': f'stop{len(self.stops)+1}',
                               'line': list(self.pending)})
            self.pending = []
            self.msg = 'stop line added'

        elif self.mode == MODE_HOM:
            if len(self.pending) != 4:
                self.msg = 'homography needs exactly 4 points'
                return
            self.hom_src = list(self.pending)
            self.pending = []
            print('\nEnter the REAL-WORLD position of each clicked point, in metres.')
            print('Use any consistent origin -- e.g. a 3.5m lane over 20m of road:')
            print('  0 0   /   3.5 0   /   3.5 20   /   0 20\n')
            self.hom_dst = []
            for i, p in enumerate(self.hom_src, 1):
                while True:
                    try:
                        raw = input(f'  point {i} at pixel {p} -> "x y" metres: ').split()
                        self.hom_dst.append((float(raw[0]), float(raw[1])))
                        break
                    except Exception:
                        print('    enter two numbers, e.g.  3.5 20')
            self.msg = 'homography set'

    def _undo(self):
        if self.pending:
            self.pending.pop()
        elif self.awaiting_heading is not None:
            self.awaiting_heading = None
            self.msg = 'cancelled lane'
        elif self.mode == MODE_LANE and self.lanes:
            self.lanes.pop()
        elif self.mode == MODE_STOP and self.stops:
            self.stops.pop()
        elif self.mode == MODE_HOM:
            self.hom_src, self.hom_dst = [], []

    # -- drawing -----------------------------------------------------------

    def render(self):
        img = self.base.copy()
        overlay = img.copy()

        for l in self.lanes:
            pts = np.array(l['polygon'], np.int32)
            cv2.fillPoly(overlay, [pts], (0, 120, 0))
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            cx = int(sum(p[0] for p in l['polygon']) / len(l['polygon']))
            cy = int(sum(p[1] for p in l['polygon']) / len(l['polygon']))
            hx, hy = l['heading']
            cv2.arrowedLine(img, (cx, cy), (int(cx + hx * 70), int(cy + hy * 70)),
                            (0, 255, 255), 3, tipLength=.35)
            label = l['name'] + (f" [{l['approach']}]" if l.get('approach') else '')
            cv2.putText(img, label, (cx - 20, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

        cv2.addWeighted(overlay, .25, img, .75, 0, img)

        for s in self.stops:
            (x1, y1), (x2, y2) = s['line']
            cv2.line(img, (x1, y1), (x2, y2), (0, 165, 255), 3)
            cv2.putText(img, s['name'], (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 165, 255), 2)

        for i, p in enumerate(self.hom_src):
            cv2.circle(img, p, 7, (255, 120, 0), -1)
            cv2.putText(img, str(i + 1), (p[0] + 9, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 120, 0), 2)

        for p in self.pending:
            cv2.circle(img, p, 5, COL[self.mode], -1)
        if len(self.pending) > 1:
            cv2.polylines(img, [np.array(self.pending, np.int32)], False,
                          COL[self.mode], 2)

        bar = f'MODE:{self.mode.upper()}  lanes:{len(self.lanes)}  stops:{len(self.stops)}  ' \
              f'hom:{"set" if self.hom_src else "-"}   [L]ane [S]top [H]om  [D]one [U]ndo [W]rite [Q]uit'
        cv2.rectangle(img, (0, 0), (self.w, 30), (0, 0, 0), -1)
        cv2.putText(img, bar, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        if self.msg:
            cv2.rectangle(img, (0, self.h - 26), (self.w, self.h), (0, 0, 0), -1)
            cv2.putText(img, self.msg, (8, self.h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
        return img

    def to_calibration(self) -> dict:
        d: dict = {'lanes': [{'name': l['name'],
                              'polygon': [[int(x), int(y)] for x, y in l['polygon']],
                              'heading': l['heading'],
                              **({'approach': l['approach']} if l.get('approach') else {})}
                             for l in self.lanes]}
        if self.stops:
            d['stop_lines'] = [{'name': s['name'],
                                'line': [[int(x), int(y)] for x, y in s['line']]}
                               for s in self.stops]
        if self.hom_src and len(self.hom_dst) == 4:
            d['homography'] = {'src': [[int(x), int(y)] for x, y in self.hom_src],
                               'dst_m': [[float(a), float(b)] for a, b in self.hom_dst]}
        return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='video (a frame is grabbed from it)')
    ap.add_argument('--config', default='config/cam_demo.yaml')
    ap.add_argument('--frame', type=int, default=None, help='frame index to draw on')
    args = ap.parse_args()

    from senti.ingest.source import resolve_source
    cap = cv2.VideoCapture(resolve_source(args.source))
    if not cap.isOpened():
        raise SystemExit(f'cannot open {args.source}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = args.frame if args.frame is not None else max(0, total // 3)

    def grab(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(i, max(0, total - 1))))
        ok, f = cap.read()
        return f if ok else None

    frame = grab(idx)
    if frame is None:
        raise SystemExit('could not read a frame')

    cal = Calibrator(frame)
    win = 'SPEC Traffic AI -- calibration'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1280, cal.w), min(760, cal.h))
    cv2.setMouseCallback(win, cal.on_mouse)

    print(__doc__)
    while True:
        cv2.imshow(win, cal.render())
        k = cv2.waitKey(20) & 0xFF
        if k in (ord('q'), 27):
            print('quit without saving'); break
        elif k == ord('l'): cal.mode = MODE_LANE;  cal.pending = []
        elif k == ord('s'): cal.mode = MODE_STOP;  cal.pending = []
        elif k == ord('h'): cal.mode = MODE_HOM;   cal.pending = []
        elif k == ord('d'): cal._finish()
        elif k == ord('u'): cal._undo()
        elif k == ord('c'):
            cal.lanes.clear(); cal.stops.clear(); cal.pending.clear()
            cal.hom_src, cal.hom_dst = [], []; cal.msg = 'cleared'
        elif k in (ord('['), ord(']')):
            idx += -30 if k == ord('[') else 30
            f = grab(idx)
            if f is not None:
                cal.base = f; cal.msg = f'frame {idx}'
        elif k == ord('w'):
            if not cal.lanes:
                cal.msg = 'draw at least one lane before writing'
                continue
            p = Path(args.config)
            cfg = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
            cfg['calibration'] = cal.to_calibration()
            p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
            print(f'\nwrote calibration -> {p}')
            print(f'  {len(cal.lanes)} lane(s), {len(cal.stops)} stop line(s), '
                  f'homography {"yes" if cal.hom_src else "no"}')
            cal.msg = f'saved to {p.name}'
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
