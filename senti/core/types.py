"""
senti.core.types
================
The contract between perception and everything else.

Rules, the evidence buffer, the portal and the challan generator all consume
these objects. None of them imports a model, a tensor, or supervision. That is
the point: the detector can be swapped -- different weights, different
framework, even a hardware sensor -- without a single downstream file changing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --- DriveIndia taxonomy ----------------------------------------------------
# Ids inferred from the DriveIndia paper's class ordering, cross-checked against
# observed label frequencies; the release ships no class-name file.
CLASS_NAMES: dict[int, str] = {
    0: 'pedestrian',            1: 'bicycle',              2: 'car',
    3: 'motorcycle',            4: 'route_board',          5: 'bus',
    6: 'commercial_vehicle',    7: 'truck',                8: 'traffic_sign',
    9: 'traffic_light',         10: 'auto_rickshaw',       11: 'ambulance',
    12: 'construction_vehicle', 13: 'animal',              14: 'unmarked_speed_bump',
    15: 'marked_speed_bump',    16: 'pothole',             17: 'police_vehicle',
    18: 'tractor',              19: 'pushcart',            20: 'temp_traffic_barrier',
    21: 'rumble_strips',        22: 'traffic_cone',        23: 'zebra_crossing',
    24: 'undocumented_24',      25: 'undocumented_25',     26: 'undocumented_26',
    27: 'undocumented_27',
}

# Zero or near-zero support in the released subset. The model emits them but
# never learned them -- never build a rule on one of these.
UNRELIABLE: frozenset[str] = frozenset({
    'ambulance', 'animal', 'police_vehicle', 'pothole', 'pushcart',
    'rumble_strips', 'undocumented_26', 'undocumented_27',
})

VEHICLE_CLASSES = frozenset({
    'bicycle', 'car', 'motorcycle', 'bus', 'commercial_vehicle',
    'truck', 'auto_rickshaw', 'tractor', 'construction_vehicle',
})
TWO_WHEELER_CLASSES = frozenset({'motorcycle', 'bicycle'})
THREE_WHEELER_CLASSES = frozenset({'auto_rickshaw'})

# Passenger Car Units (IRC). Indian traffic is heterogeneous -- an approach
# holding 40 motorcycles and one holding 40 buses impose very different demand.
# Counting vehicles treats them identically; counting PCU does not.
PCU: dict[str, float] = {
    'bicycle': 0.4, 'motorcycle': 0.5, 'auto_rickshaw': 0.8, 'car': 1.0,
    'tractor': 1.5, 'commercial_vehicle': 2.2, 'truck': 2.2, 'bus': 3.0,
    'construction_vehicle': 3.0,
}


@dataclass
class Detection:
    """One detected object in one frame."""

    cls_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    track_id: Optional[int] = None
    frame_index: int = -1

    # -- geometry the rule engine leans on ---------------------------------

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Where the object meets the road.

        Rules test THIS against stop lines and lane polygons, never the box
        centre: a bus's centre floats metres above the tarmac and would cross a
        painted line long before its wheels do. It is also the only point that
        lies on the road plane, so it is the correct input to a homography.
        """
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_vehicle(self) -> bool:
        return self.cls_name in VEHICLE_CLASSES

    @property
    def is_two_wheeler(self) -> bool:
        return self.cls_name in TWO_WHEELER_CLASSES

    @property
    def is_three_wheeler(self) -> bool:
        return self.cls_name in THREE_WHEELER_CLASSES

    @property
    def is_reliable(self) -> bool:
        return self.cls_name not in UNRELIABLE

    @property
    def pcu(self) -> float:
        return PCU.get(self.cls_name, 0.0)

    def iou(self, other: 'Detection') -> float:
        ax1, ay1, ax2, ay2 = self.xyxy
        bx1, by1, bx2, by2 = other.xyxy
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def overlap_fraction(self, other: 'Detection') -> float:
        """How much of THIS box lies inside `other`.

        Asymmetric on purpose. A rider box is small and a motorcycle box is
        large, so IoU is near-useless for associating them, while "what
        fraction of the rider sits on the bike" is exactly the right question.
        """
        ax1, ay1, ax2, ay2 = self.xyxy
        bx1, by1, bx2, by2 = other.xyxy
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        return (iw * ih) / self.area if self.area > 0 else 0.0


@dataclass
class FrameResult:
    """Everything the perception layer knows about a single frame."""

    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    timestamp: float = 0.0          # seconds into the stream

    def of_class(self, *names: str) -> list[Detection]:
        wanted = set(names)
        return [d for d in self.detections if d.cls_name in wanted]

    def by_track(self, track_id: int) -> Optional[Detection]:
        for d in self.detections:
            if d.track_id == track_id:
                return d
        return None

    @property
    def vehicles(self) -> list[Detection]:
        return [d for d in self.detections if d.is_vehicle]

    @property
    def two_wheelers(self) -> list[Detection]:
        return [d for d in self.detections if d.is_two_wheeler]

    @property
    def three_wheelers(self) -> list[Detection]:
        return [d for d in self.detections if d.is_three_wheeler]

    @property
    def pedestrians(self) -> list[Detection]:
        return self.of_class('pedestrian')

    @property
    def traffic_lights(self) -> list[Detection]:
        return self.of_class('traffic_light')

    @property
    def total_pcu(self) -> float:
        """PCU-weighted demand -- the input to adaptive signal control."""
        return sum(d.pcu for d in self.detections)

    def riders_on(self, vehicle: Detection, min_overlap: float = 0.35) -> list[Detection]:
        """Pedestrian boxes sitting on a vehicle -- the basis of triple riding.

        Returns CANDIDATES, not a verdict. A single frame will occasionally
        catch a pedestrian walking behind a bike; requiring the condition to
        hold across frames is the Rule base class's job.
        """
        return [p for p in self.pedestrians
                if p.overlap_fraction(vehicle) >= min_overlap]
