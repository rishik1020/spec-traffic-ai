"""
senti.signal
============
Adaptive signal timing: measure congestion per junction arm, recommend a plan.

This package is the project's second output type. Everything under `senti.rules`
produces a Violation about one vehicle; this produces a timing RECOMMENDATION
about a whole junction, from the same single perception pass. The detector runs
once and both consumers read the same FrameResult.

It is advisory by design and cannot actuate a signal head -- see
`controller.AdaptiveController` for why that boundary is deliberate rather than
merely unfinished.
"""

from .controller import AdaptiveController, Phase, SignalPlan
from .demand import ApproachDemand, DemandMeter

__all__ = ['AdaptiveController', 'ApproachDemand', 'DemandMeter', 'Phase',
           'SignalPlan']
