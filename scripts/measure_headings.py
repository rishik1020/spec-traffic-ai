"""
measure_headings.py
===================
Work out which way traffic actually flows in each lane, from the footage.

WHY GUESSING THE HEADING IS THE BIGGEST FALSE-POSITIVE SOURCE LEFT
`wrong_way` compares a vehicle's direction of travel against the direction its
lane is permitted to flow. That permitted direction is hand-entered during
calibration, and if it is entered backwards the rule inverts: every lawful
vehicle is flagged and the actual offenders are the only ones ignored. The rule
cannot detect this. It has no way to know the number it was given is wrong, and
it will fire confidently on a hundred innocent drivers.

Clicking an arrow on a still frame is a guess. A still frame does not show
motion, and on a divided road with traffic in both carriageways the two
directions look identical in a photograph.

WHAT THIS DOES INSTEAD
Tracks vehicles for a while, computes each track's net direction of travel, and
reports the dominant heading inside each declared lane. The permitted direction
of a lane is, to a very good approximation, the direction the overwhelming
majority of its traffic is already going.

It also reports how DOMINANT that majority is, which is the part worth reading.
A lane where 95% of tracks agree is a one-way lane and the heading is settled. A
lane split 60/40 is not a lane -- it is a polygon drawn across a median, or a
turning box, or two lanes treated as one, and no single heading will describe
it. In that case the right answer is to redraw the polygon, not to pick the
winner.

THE ASSUMPTION, STATED
Most traffic obeys the road. If a majority of vehicles in a lane are genuinely
driving the wrong way, this will confidently learn the wrong direction. That is
acceptable for calibration on ordinary footage and must not be run on a clip
chosen because it contains violations.

USAGE
    python scripts/measure_headings.py --source data/videos/kolkata.mp4 \
        --config config/cam_demo.yaml
    python scripts/measure_headings.py --source clip.mp4 --config cam.yaml --apply
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from senti.calibration import Calibration          # noqa: E402
from senti.ingest.source import VideoSource        # noqa: E402
from senti.perception.detector import TrafficDetector   # noqa: E402

COMPASS = [((0, -1), 'up the frame'), ((0, 1), 'down the frame'),
           ((1, 0), 'left to right'), ((-1, 0), 'right to left'),
           ((0.707, -0.707), 'up-right'), ((-0.707, -0.707), 'up-left'),
           ((0.707, 0.707), 'down-right'), ((-0.707, 0.707), 'down-left')]


def describe(v: tuple[float, float]) -> str:
    best, name = -2.0, '?'
    for ref, label in COMPASS:
        dot = v[0] * ref[0] + v[1] * ref[1]
        if dot > best:
            best, name = dot, label
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--source', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--weights', default='runs/driveindia/weights/best.pt')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--min-displacement', type=float, default=35.0,
                    help='pixels a track must cover before its heading counts')
    ap.add_argument('--apply', action='store_true',
                    help='write the measured headings back into the config')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}
    cal = Calibration.from_dict(cfg.get('calibration'))
    if not cal.lanes:
        raise SystemExit(f'{args.config} has no lanes. Draw them first:\n'
                         f'    python scripts/calibrate.py --source {args.source} '
                         f'--config {args.config}')

    det = TrafficDetector(weights=args.weights, imgsz=args.imgsz, device=args.device)
    src = VideoSource(args.source, stride=args.stride, max_frames=args.max_frames)
    info = src.open()
    print(f'[measure] {info.width}x{info.height} @ {info.fps:.1f}fps, '
          f'{len(cal.lanes)} lane(s)')

    # first and last seen position of each track, plus where it spent its time
    first: dict[int, tuple] = {}
    last: dict[int, tuple] = {}
    lane_votes: dict[int, defaultdict] = defaultdict(lambda: defaultdict(int))

    for idx, ts, frame in src.frames():
        r = det.track(frame, frame_index=idx, timestamp=ts)
        for d in r.detections:
            if d.track_id is None or not d.is_vehicle or not d.is_reliable:
                continue
            pt = d.bottom_center
            first.setdefault(d.track_id, pt)
            last[d.track_id] = pt
            lane = cal.lane_at(pt)
            if lane is not None:
                lane_votes[d.track_id][lane.name] += 1
        if idx % 150 == 0 and idx:
            print(f'  frame {idx}: {len(first)} track(s) seen')

    # assign each track to the lane it spent the most frames in
    per_lane: dict[str, list] = defaultdict(list)
    short = 0
    for tid, start in first.items():
        end = last.get(tid)
        if end is None or not lane_votes.get(tid):
            continue
        dx, dy = end[0] - start[0], end[1] - start[1]
        mag = math.hypot(dx, dy)
        if mag < args.min_displacement:
            short += 1                 # parked, or only briefly seen
            continue
        lane = max(lane_votes[tid].items(), key=lambda kv: kv[1])[0]
        per_lane[lane].append((dx / mag, dy / mag))

    print(f'\n[measure] {len(first)} tracks, {short} too stationary to use\n')
    print('=' * 72)

    suggestions = {}
    for lane in cal.lanes:
        vecs = per_lane.get(lane.name, [])
        print(f'\nLANE "{lane.name}"   declared heading '
              f'[{lane.heading[0]:+.2f}, {lane.heading[1]:+.2f}] '
              f'({describe(lane.heading)})')
        if len(vecs) < 3:
            print(f'  only {len(vecs)} usable track(s) -- not enough to judge. '
                  f'Longer footage, or the polygon may be in the wrong place.')
            continue

        mx = sum(v[0] for v in vecs) / len(vecs)
        my = sum(v[1] for v in vecs) / len(vecs)
        mag = math.hypot(mx, my)
        mean = (mx / mag, my / mag) if mag > 1e-6 else (0.0, -1.0)

        # How much the traffic agrees. The mean of unit vectors has length 1
        # only when every track points the same way, and collapses toward 0 as
        # they cancel -- so its magnitude IS the consistency.
        agree = sum(1 for v in vecs
                    if v[0] * mean[0] + v[1] * mean[1] > 0) / len(vecs)
        against = sum(1 for v in vecs
                      if v[0] * lane.heading[0] + v[1] * lane.heading[1] < 0)

        print(f'  {len(vecs)} tracks  |  measured [{mean[0]:+.2f}, {mean[1]:+.2f}] '
              f'({describe(mean)})')
        print(f'  coherence {mag:.2f}   {100*agree:.0f}% of tracks go that way')
        print(f'  {against} track(s) oppose the DECLARED heading '
              f'({100*against/len(vecs):.0f}%)')

        dot = mean[0] * lane.heading[0] + mean[1] * lane.heading[1]
        if mag < 0.55:
            print('  >> INCOHERENT. Traffic here does not share a direction, so no '
                  'single heading describes it.')
            print('     The polygon probably spans a median, a junction box, or '
                  'two opposing lanes. Redraw it rather than picking a winner.')
        elif dot < -0.5:
            print('  >> DECLARED HEADING IS BACKWARDS. Every lawful vehicle in '
                  'this lane is being flagged.')
            suggestions[lane.name] = mean
        elif dot < 0.7:
            print(f'  >> declared heading is {math.degrees(math.acos(max(-1, min(1, dot)))):.0f} '
                  f'degrees off the measured flow. Worth correcting.')
            suggestions[lane.name] = mean
        else:
            print('  >> declared heading agrees with the traffic.')

    print('\n' + '=' * 72)
    if not suggestions:
        print('No changes suggested.')
        return

    print('SUGGESTED headings:')
    for name, v in suggestions.items():
        print(f'  {name}: [{v[0]:+.3f}, {v[1]:+.3f}]   ({describe(v)})')

    if not args.apply:
        print('\nRe-run with --apply to write these into the config.')
        return

    lanes_cfg = cfg.setdefault('calibration', {}).setdefault('lanes', [])
    for row in lanes_cfg:
        if row.get('name') in suggestions:
            v = suggestions[row['name']]
            row['heading'] = [round(v[0], 3), round(v[1], 3)]
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    print(f'\nwrote measured headings into {cfg_path}')


if __name__ == '__main__':
    main()
