"""
senti.rules.triple_riding
=========================
More than two people on a two-wheeler.

WHY THIS RULE IS CHEAP AND THE OTHERS ARE NOT
It needs no calibration, no signal state and no homography. The detector already
emits `motorcycle` and `pedestrian`; a rider is simply a pedestrian box sitting
on a motorcycle box. That makes this the only junction-grade violation that runs
on ANY camera the moment the model loads -- including footage recorded before
anyone thought about geometry.

WHY OVERLAP FRACTION AND NOT IoU
A rider's box is small and the motorcycle's is large, so their IoU is tiny even
when the rider is unmistakably on the bike. The right question is asymmetric --
"what fraction of the RIDER lies inside the motorcycle box" -- which is exactly
what `Detection.overlap_fraction` computes.

THE TWO FALSE POSITIVES THIS GUARDS AGAINST
1. A pedestrian walking behind a bike briefly overlaps it. Defeated by temporal
   consistency: the base class requires the count to hold for N consecutive
   frames on the same track, and a pedestrian walking past does not.
2. A rider on the NEXT bike over, in dense traffic, overlapping this one's box.
   Defeated geometrically: a rider's feet cannot be below the bike's wheels, so
   candidates whose contact point sits materially lower than the motorcycle's
   are discarded.

WHY IT UNDER-COUNTS, AND WHY THAT IS CORRECT
The third rider is usually the most occluded and is the one the detector misses,
so this rule fires less often than the offence occurs. That is the safe
direction of error for a system that ends in a fine: a missed violation costs
nothing, a wrongful challan costs a person money and an afternoon.

LEGAL BASIS
Section 128 limits a two-wheeler to the driver and one pillion rider; Section
194C penalises carrying more. Evidence must show the riders, so the evidence
package records how many were counted and how confidently.
"""

from __future__ import annotations

from typing import Optional

from ..core.types import Detection, FrameResult
from .base import Rule


class TripleRidingRule(Rule):
    name = 'triple_riding'
    applies_to = ('motorcycle',)
    requires = ()                 # nothing: no calibration, no signal
    stateful = False
    min_frames = 12               # ~0.5s at 25fps -- survives a passer-by
    cooldown_frames = 250

    mv_act_section = 'MV Act s.128 r/w s.194C (more than one pillion rider)'
    description = 'More than two persons riding a two-wheeler'

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)

        # How much of a person must lie inside the motorcycle box to count as a
        # rider. 0.35 is deliberately forgiving: a rider's head and shoulders
        # often sit above the bike's box entirely.
        self.min_overlap = float(self.config.get('min_overlap', 0.35))

        # The offence. Three is the legal threshold, not a tuning knob.
        self.max_riders = int(self.config.get('max_riders', 2))

        # A rider's contact point cannot be far below the motorcycle's. Measured
        # as a fraction of the bike's own box height so it is scale-free: the
        # same test works on a bike 10m away and one 100m away.
        self.foot_tolerance = float(self.config.get('foot_tolerance', 0.35))

        # Riders on a bike are small and often partly occluded, so they arrive
        # with lower confidence than a clear pedestrian. Too high a floor here
        # removes exactly the third rider the rule exists to find.
        self.min_rider_conf = float(self.config.get('min_rider_conf', 0.25))

    # ---------------------------------------------------------------------

    def _riders(self, bike: Detection, result: FrameResult) -> list[Detection]:
        """People on this motorcycle, after the geometric sanity checks."""
        bike_bottom = bike.bottom_center[1]
        limit = bike_bottom + self.foot_tolerance * bike.height

        out = []
        for p in result.pedestrians:
            if p.confidence < self.min_rider_conf:
                continue
            if p.overlap_fraction(bike) < self.min_overlap:
                continue
            # feet below the bike's wheels -> standing beside it, not on it
            if p.bottom_center[1] > limit:
                continue
            out.append(p)
        return out

    def evaluate(self, det: Detection, result: FrameResult,
                 context: dict) -> Optional[tuple[str, dict]]:
        riders = self._riders(det, result)
        count = len(riders)

        if count <= self.max_riders:
            return None

        overlaps = sorted(r.overlap_fraction(det) for r in riders)
        confs = sorted(r.confidence for r in riders)

        reason = (
            f'{count} persons detected on motorcycle (track #{det.track_id}); '
            f'the Motor Vehicles Act permits a driver and one pillion rider'
        )

        # margin_norm feeds the Evidence Defensibility Score. A bike carrying
        # four is stronger evidence than one carrying three, but the weakest
        # rider's overlap matters more: the whole finding rests on whether that
        # marginal person is really on the bike.
        excess = (count - self.max_riders - 1) / 2.0
        weakest = overlaps[0]
        margin = 0.5 * min(1.0, excess + 0.5) + 0.5 * min(1.0, weakest / 0.8)

        return reason, {
            'rider_count': count,
            'permitted': self.max_riders,
            'rider_overlaps': [round(o, 3) for o in overlaps],
            'rider_confidences': [round(c, 3) for c in confs],
            'weakest_overlap': round(weakest, 3),
            'weakest_rider_conf': round(confs[0], 3),
            'margin_norm': round(max(0.0, min(1.0, margin)), 3),
        }
