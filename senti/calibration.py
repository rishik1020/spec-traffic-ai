"""
senti.calibration
=================
Per-camera geometry: lanes, stop lines, and the pixel-to-metre mapping.

WHY THIS EXISTS
A camera without calibration cannot support a single geometric rule. Not
"performs worse" -- cannot. A stop line is a stop line only because someone
said where it is; "wrong way" means nothing until the permitted direction of
each lane is declared.

The first live test made this concrete: a global `allowed_heading` on a two-way
road flagged eight vehicles travelling perfectly legally, because ONE direction
per camera cannot describe a road where traffic moves both ways. No value of
that setting fixes it. Direction has to be per LANE.

THE FALSE-POSITIVE FIX
`lane_at()` returns None for anything outside every defined lane, and rules are
required to abstain rather than guess. An uncalibrated region produces NO
verdict. That is the correct default for a system that issues fines: silence is
cheap, a wrongful challan is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

Point = tuple[float, float]


def _unit(v: Sequence[float]) -> tuple[float, float]:
    mag = math.hypot(v[0], v[1]) or 1.0
    return (v[0] / mag, v[1] / mag)


@dataclass
class Lane:
    """A drivable region plus the direction traffic is permitted to travel.

    Headings are in IMAGE coordinates, where y increases DOWNWARD:
        [0, -1] -> toward the top of frame
        [0,  1] -> toward the bottom
        [1,  0] -> left to right
    """

    name: str
    polygon: list[Point]
    heading: tuple[float, float]
    speed_limit_kmph: Optional[float] = None
    # India drives on the left, so the fast lane is the RIGHTMOST one. Declared
    # per camera rather than inferred from position -- camera angle makes
    # "rightmost in frame" unreliable.
    fast_lane: bool = False

    def __post_init__(self):
        self.heading = _unit(self.heading)
        self._np = np.array(self.polygon, dtype=np.int32)

    def contains(self, pt: Point) -> bool:
        import cv2
        return cv2.pointPolygonTest(self._np, (float(pt[0]), float(pt[1])), False) >= 0

    def angle_from_heading(self, v: Sequence[float]) -> float:
        """Degrees between a travel vector and this lane's permitted direction."""
        u = _unit(v)
        dot = max(-1.0, min(1.0, u[0] * self.heading[0] + u[1] * self.heading[1]))
        return math.degrees(math.acos(dot))


@dataclass
class StopLine:
    """A line vehicles must not cross while the signal is red."""

    name: str
    line: tuple[Point, Point]
    lanes: list[str] = field(default_factory=list)   # empty = applies to all

    def side(self, pt: Point) -> float:
        """Signed side of the line. Sign flips when a point crosses it.

        This is the 2D cross product of the line vector with the point offset --
        positive on one side, negative on the other, zero exactly on it. A
        crossing is therefore just a sign change between two frames, which needs
        no thresholds and no tuning.
        """
        (x1, y1), (x2, y2) = self.line
        return (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)

    def crossed(self, prev: Point, curr: Point) -> bool:
        a, b = self.side(prev), self.side(curr)
        return a != 0 and b != 0 and (a > 0) != (b > 0)


@dataclass
class Homography:
    """Pixel -> real-world metres on the road plane.

    Only valid for points ON the road surface, which is why rules must use
    `Detection.bottom_center` (where the tyres meet the tarmac) and never the
    box centre, which floats above the plane.
    """

    src: list[Point]                  # 4 image points
    dst_m: list[Point]                # the same 4 points in metres

    def __post_init__(self):
        import cv2
        self._H, _ = cv2.findHomography(
            np.array(self.src, dtype=np.float32),
            np.array(self.dst_m, dtype=np.float32))

    def to_metres(self, pt: Point) -> tuple[float, float]:
        p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
        import cv2
        out = cv2.perspectiveTransform(p, self._H)[0][0]
        return float(out[0]), float(out[1])

    def distance_m(self, a: Point, b: Point) -> float:
        ax, ay = self.to_metres(a)
        bx, by = self.to_metres(b)
        return math.hypot(bx - ax, by - ay)


@dataclass
class Calibration:
    lanes: list[Lane] = field(default_factory=list)
    stop_lines: list[StopLine] = field(default_factory=list)
    homography: Optional[Homography] = None

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> 'Calibration':
        cfg = cfg or {}
        lanes = [
            Lane(name=l.get('name', f'lane{i}'),
                 polygon=[tuple(p) for p in l['polygon']],
                 heading=tuple(l.get('heading', [0, -1])),
                 speed_limit_kmph=l.get('speed_limit_kmph'),
                 fast_lane=bool(l.get('fast_lane', False)))
            for i, l in enumerate(cfg.get('lanes', []) or [])
        ]
        stops = [
            StopLine(name=s.get('name', f'stop{i}'),
                     line=(tuple(s['line'][0]), tuple(s['line'][1])),
                     lanes=s.get('lanes', []) or [])
            for i, s in enumerate(cfg.get('stop_lines', []) or [])
        ]
        h = cfg.get('homography')
        hom = Homography(src=[tuple(p) for p in h['src']],
                         dst_m=[tuple(p) for p in h['dst_m']]) if h else None
        return cls(lanes=lanes, stop_lines=stops, homography=hom)

    def to_dict(self) -> dict:
        d: dict = {'lanes': [
            {'name': l.name,
             'polygon': [[int(x), int(y)] for x, y in l.polygon],
             'heading': [round(l.heading[0], 3), round(l.heading[1], 3)],
             **({'speed_limit_kmph': l.speed_limit_kmph} if l.speed_limit_kmph else {}),
             **({'fast_lane': True} if l.fast_lane else {})}
            for l in self.lanes
        ]}
        if self.stop_lines:
            d['stop_lines'] = [
                {'name': s.name,
                 'line': [[int(s.line[0][0]), int(s.line[0][1])],
                          [int(s.line[1][0]), int(s.line[1][1])]],
                 **({'lanes': s.lanes} if s.lanes else {})}
                for s in self.stop_lines
            ]
        if self.homography:
            d['homography'] = {
                'src': [[int(x), int(y)] for x, y in self.homography.src],
                'dst_m': [[float(x), float(y)] for x, y in self.homography.dst_m],
            }
        return d

    # -- queries used by rules --------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return bool(self.lanes)

    def lane_at(self, pt: Point) -> Optional[Lane]:
        """Which lane contains this point, or None.

        None is meaningful: it means "outside every declared lane", and every
        rule must treat that as ABSTAIN, never as a violation.
        """
        for lane in self.lanes:
            if lane.contains(pt):
                return lane
        return None

    def stop_lines_for(self, lane_name: Optional[str]) -> list[StopLine]:
        return [s for s in self.stop_lines
                if not s.lanes or (lane_name and lane_name in s.lanes)]
