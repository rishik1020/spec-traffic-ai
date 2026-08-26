"""
senti.signal.controller
=======================
Turn measured demand into a recommended signal plan.

READ THIS FIRST -- IT RECOMMENDS, IT DOES NOT CONTROL
This module never actuates a signal head and there is no code path by which it
could. A traffic signal is safety-critical hardware: a bad plan does not
produce a wrong challan, it produces a collision. What comes out of here is a
timing recommendation plus the arithmetic that produced it, written to a log a
traffic engineer can audit and a simulator can replay. Deploying it for real
means handing that plan to the junction's existing controller through whatever
interface its vendor provides, with an engineer signing off -- which is exactly
how commercial adaptive systems are commissioned.

WHY WEBSTER AND NOT A NEURAL NETWORK
Webster's method (1958) is still the basis of signal design worldwide, IRC:93
included. It is four lines of arithmetic, every term has a physical meaning,
and a traffic engineer can check it by hand. A learned controller would have to
justify itself to the same engineer with no way to show its working. The
project's whole argument is that enforcement and control decisions have to
survive being questioned, so the explainable method wins on principle here, not
merely on convenience.

    y_i = q_i / s_i           demand flow / saturation flow, per phase
    Y   = sum of the critical y_i
    L   = lost time: intergreen + start-up, once per phase per cycle
    C   = (1.5 L + 5) / (1 - Y)          optimum cycle
    g_i = (y_i / Y) x (C - L)            green, split by demand share

WHERE WEBSTER BREAKS, AND WHAT HAPPENS THEN
As Y approaches 1 the cycle goes to infinity: the formula is telling you the
junction is over capacity and no timing can fix it. Peak-hour Indian junctions
sit there routinely. Above `max_practical_y` the controller stops pretending,
caps the cycle, and splits the green by QUEUE share instead of flow share --
which is the right objective once you have already lost: clear the longest
queue first. The plan says which mode produced it.

THE CONSTRAINTS ARE NOT TUNING KNOBS
`min_green` exists because a pedestrian who stepped off the kerb must reach the
other side. `max_green` and the fixed phase order exist so a quiet arm is still
served every cycle -- an optimiser left alone will starve it. `max_delta_s`
exists because drivers learn a junction's rhythm and a plan that lurches
between cycles causes the late-amber running the system is meant to prevent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Optional

from .demand import ApproachDemand


@dataclass
class Phase:
    """A set of arms that get green together.

    Opposing arms usually move at the same time -- north and south run while
    east and west wait -- so demand has to be combined per PHASE, not per arm,
    and the phase is sized by its CRITICAL (busiest) arm. Sizing by the average
    would under-serve the busy one every cycle.
    """

    name: str
    approaches: tuple[str, ...]

    @classmethod
    def from_dict(cls, d: dict, index: int) -> 'Phase':
        return cls(name=d.get('name', f'phase{index + 1}'),
                   approaches=tuple(d.get('approaches', []) or []))


@dataclass
class SignalPlan:
    """A recommended timing plan, with its own reasoning attached."""

    timestamp: float
    mode: str                                    # fixed | webster | oversaturated
    cycle_s: float
    order: list[str]                             # phase order, one entry per service
    greens: dict[str, float]                     # phase -> total green per cycle
    services: dict[str, int]                     # phase -> services per cycle
    lost_time_s: float
    intergreen_s: float
    reason: str                                  # plain language, for the engineer
    demand: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    changed: bool = True

    @property
    def confidence(self) -> str:
        return 'low' if self.caveats else 'ok'

    def to_dict(self) -> dict:
        return {
            'timestamp': round(self.timestamp, 2),
            'mode': self.mode,
            'cycle_s': round(self.cycle_s, 1),
            'order': self.order,
            'greens_s': {k: round(v, 1) for k, v in self.greens.items()},
            'services': self.services,
            'lost_time_s': round(self.lost_time_s, 1),
            'intergreen_s': self.intergreen_s,
            'reason': self.reason,
            'confidence': self.confidence,
            'caveats': self.caveats,
            'demand': self.demand,
            'advisory_only': True,
        }

    def summary(self) -> str:
        parts = ' | '.join(f'{k} {v:.0f}s' + (f' x{self.services[k]}'
                                              if self.services.get(k, 1) > 1 else '')
                           for k, v in self.greens.items())
        return f'cycle {self.cycle_s:.0f}s [{self.mode}] -> {parts}'


class AdaptiveController:
    """Demand in, recommended plan out, on a fixed update interval.

    Deliberately memoryless apart from the previous plan, which it needs for
    rate limiting. Everything else is recomputed from the current reading, so a
    replay of the demand log reproduces the plan exactly.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}

        self.phases: list[Phase] = [Phase.from_dict(p, i)
                                    for i, p in enumerate(cfg.get('phases') or [])]

        # Saturation flow: PCU/hr an arm discharges while it has green. IRC
        # puts a straight-through urban lane near 1800 PCU/hr; it drops with
        # narrow lanes, parking, and heavy turning movements, so it is
        # configurable per arm and should be MEASURED at a real junction rather
        # than taken from a table.
        sat = cfg.get('saturation_flow_pcu_hr', 1800.0)
        self.saturation: dict[str, float] = sat if isinstance(sat, dict) else {}
        self.saturation_default = float(sat) if not isinstance(sat, dict) else \
            float(cfg.get('saturation_flow_default', 1800.0))

        # Lost time. Intergreen = amber + all-red, during which nobody moves.
        # Start-up lost time is the couple of seconds the front of the queue
        # takes to get going after green. Both are per phase SERVICE.
        self.intergreen_s = float(cfg.get('intergreen_s', 5.0))
        self.startup_lost_s = float(cfg.get('startup_lost_s', 2.0))

        mg = cfg.get('min_green_s', 12.0)
        self.min_green: dict[str, float] = mg if isinstance(mg, dict) else {}
        self.min_green_default = float(mg) if not isinstance(mg, dict) else \
            float(cfg.get('min_green_default', 12.0))
        self.max_green_s = float(cfg.get('max_green_s', 60.0))

        self.min_cycle_s = float(cfg.get('min_cycle_s', 40.0))
        self.max_cycle_s = float(cfg.get('max_cycle_s', 120.0))
        self.max_practical_y = float(cfg.get('max_practical_y', 0.90))
        self.max_delta_s = float(cfg.get('max_delta_s', 10.0))

        self.update_every_s = float(cfg.get('update_every_s', 60.0))

        # Peak detection. Clock windows are what a traffic department already
        # thinks in; the demand trigger is what actually matters, because a
        # jam does not check the time. With neither configured, adapt always.
        self.peak_windows = [self._parse_window(w)
                             for w in (cfg.get('peak_windows') or [])]
        trig = cfg.get('demand_trigger_pcu')
        self.demand_trigger_pcu = float(trig) if trig is not None else None

        # Off-peak fallback: the junction's existing fixed plan. Recommending
        # nothing is not an option -- the log has to say what should be running
        # at every moment, including "leave it alone".
        self.baseline_greens: dict[str, float] = {
            k: float(v) for k, v in (cfg.get('baseline_greens_s') or {}).items()}

        # Serving one arm twice per cycle genuinely helps under severe
        # imbalance, but it costs an extra intergreen every cycle and changes
        # the rhythm drivers have learned. Off unless asked for.
        self.allow_reservice = bool(cfg.get('allow_reservice', False))
        self.reservice_ratio = float(cfg.get('reservice_ratio', 2.5))

        self._last_update: Optional[float] = None
        self._previous: Optional[SignalPlan] = None

    # -- public ------------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        return len(self.phases) >= 2

    def due(self, timestamp: float) -> bool:
        return (self._last_update is None
                or timestamp - self._last_update >= self.update_every_s)

    def update(self, demands: dict[str, ApproachDemand], timestamp: float,
               wall_clock: Optional[datetime] = None) -> Optional[SignalPlan]:
        """Recommend a plan, or None if it is not yet time to recompute."""
        if not self.is_configured or not self.due(timestamp):
            return None
        self._last_update = timestamp

        warming = [d for d in demands.values() if d.warming_up]
        if warming:
            # Not enough traffic has been watched to state a flow rate, and a
            # flow ratio built on ten seconds of arrivals is noise wearing a
            # decimal point. Recommend the existing fixed plan until the meter
            # has a real sample -- saying "keep what you have" is a valid
            # recommendation, inventing a cycle length is not.
            seen = min(d.observed_s for d in demands.values())
            plan = self._baseline_plan(
                demands, timestamp,
                f'still measuring: {seen:.0f}s observed, and a flow rate needs '
                f'a representative sample before it means anything')
        else:
            peak, why = self._is_peak(demands, wall_clock)
            plan = (self._adaptive_plan(demands, timestamp, why) if peak
                    else self._baseline_plan(demands, timestamp, why))

        plan.changed = self._materially_different(plan)
        self._previous = plan
        return plan

    # -- peak --------------------------------------------------------------

    @staticmethod
    def _parse_window(w: str) -> tuple[dtime, dtime]:
        a, b = str(w).split('-')
        hh, mm = a.strip().split(':')
        hh2, mm2 = b.strip().split(':')
        return dtime(int(hh), int(mm)), dtime(int(hh2), int(mm2))

    def _is_peak(self, demands: dict[str, ApproachDemand],
                 wall_clock: Optional[datetime]) -> tuple[bool, str]:
        if self.demand_trigger_pcu is not None:
            total = sum(d.present_pcu for d in demands.values())
            if total >= self.demand_trigger_pcu:
                return True, (f'measured demand {total:.0f} PCU is at or above '
                              f'the peak threshold of {self.demand_trigger_pcu:.0f}')
        if self.peak_windows:
            # A recorded clip has no wall clock. Falling back to "not peak"
            # would silently disable the feature on exactly the footage used to
            # demonstrate it, so the demand trigger above is the real gate and
            # this is the scheduled overlay.
            if wall_clock is not None:
                now = wall_clock.time()
                for start, end in self.peak_windows:
                    if start <= now <= end:
                        return True, (f'inside the scheduled peak window '
                                      f'{start:%H:%M}-{end:%H:%M}')
        if not self.peak_windows and self.demand_trigger_pcu is None:
            return True, 'adaptive control is unconditional on this camera'
        return False, 'off-peak: neither the clock window nor the demand threshold is met'

    # -- plans -------------------------------------------------------------

    def _baseline_plan(self, demands, timestamp: float, why: str) -> SignalPlan:
        greens = dict(self.baseline_greens) or {
            p.name: self.min_green_default * 2 for p in self.phases}
        lost = len(self.phases) * (self.intergreen_s + self.startup_lost_s)
        return SignalPlan(
            timestamp=timestamp,
            mode='fixed',
            cycle_s=sum(greens.values()) + lost,
            order=[p.name for p in self.phases],
            greens=greens,
            services={p.name: 1 for p in self.phases},
            lost_time_s=lost,
            intergreen_s=self.intergreen_s,
            reason=f'Run the existing fixed plan -- {why}.',
            demand=[d.to_dict() for d in demands.values()],
        )

    def _adaptive_plan(self, demands: dict[str, ApproachDemand],
                       timestamp: float, why: str) -> SignalPlan:
        caveats: list[str] = []

        # Each phase is sized by its CRITICAL arm -- the busiest one moving in
        # that phase. Averaging would under-serve it every single cycle.
        y: dict[str, float] = {}
        critical: dict[str, str] = {}
        queue: dict[str, float] = {}
        for ph in self.phases:
            best_y, best_arm = 0.0, ''
            q = 0.0
            for arm in ph.approaches:
                d = demands.get(arm)
                if d is None:
                    caveats.append(f'no demand reading for arm {arm!r} -- '
                                   f'it is not covered by this camera')
                    continue
                if d.truncated:
                    caveats.append(f'queue on {arm} runs past the edge of the '
                                   f'observed zone, so its demand is a FLOOR')
                s = self.saturation.get(arm, self.saturation_default)
                ratio = d.flow_pcu_hr / s if s > 0 else 0.0
                if ratio > best_y:
                    best_y, best_arm = ratio, arm
                q += d.queue_pcu
            y[ph.name] = best_y
            critical[ph.name] = best_arm
            queue[ph.name] = q

        order, services = self._service_order(queue)
        lost = len(order) * (self.intergreen_s + self.startup_lost_s)

        Y = sum(y.values())
        if Y > 2.0:
            # Y is demand divided by capacity. Above 2 the junction would be
            # carrying twice what its saturation flow says it can, which in
            # practice means the configured saturation_flow_pcu_hr is wrong for
            # this junction, not that the road is doing something remarkable.
            caveats.append(f'flow ratio Y={Y:.2f} implies demand at more than '
                           f'twice the configured saturation flow -- check '
                           f'saturation_flow_pcu_hr against a measured count')
        if Y >= self.max_practical_y:
            mode = 'oversaturated'
            cycle = self.max_cycle_s
            share = self._share(queue) or self._share(y) or \
                {p.name: 1.0 / len(self.phases) for p in self.phases}
            reason_head = (
                f'Junction is over capacity (flow ratio Y={Y:.2f}); no cycle '
                f'length can clear it. Holding the cycle at its {cycle:.0f}s '
                f'ceiling and splitting green by QUEUE share to drain the '
                f'longest queue first.')
        else:
            mode = 'webster'
            cycle = (1.5 * lost + 5.0) / (1.0 - Y) if Y > 0 else self.min_cycle_s
            cycle = min(max(cycle, self.min_cycle_s), self.max_cycle_s)
            share = self._share(y) or {p.name: 1.0 / len(self.phases)
                                       for p in self.phases}
            reason_head = (f"Webster optimum for a flow ratio of Y={Y:.2f} "
                           f"is a {cycle:.0f}s cycle.")

        effective = cycle - lost
        if effective <= 0:
            caveats.append('lost time exceeds the cycle ceiling -- too many '
                           'phase changes for this junction')
            effective = max(1.0, len(order) * self.min_green_default)

        greens = {name: share.get(name, 0.0) * effective for name in y}
        greens, cycle, clamp_notes = self._apply_limits(greens, cycle, lost, services)
        caveats.extend(clamp_notes)
        greens = self._rate_limit(greens, caveats)

        # renormalise after rate limiting so cycle == sum(green) + lost exactly
        cycle = sum(greens.values()) + lost

        return SignalPlan(
            timestamp=timestamp,
            mode=mode,
            cycle_s=cycle,
            order=order,
            greens=greens,
            services=services,
            lost_time_s=lost,
            intergreen_s=self.intergreen_s,
            reason=f'{reason_head} {self._imbalance(mode, y, queue, greens, critical)} '
                   f'Triggered because {why}.',
            demand=[d.to_dict() for d in demands.values()],
            caveats=sorted(set(caveats)),
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _share(values: dict[str, float]) -> Optional[dict[str, float]]:
        total = sum(values.values())
        if total <= 0:
            return None
        return {k: v / total for k, v in values.items()}

    def _service_order(self, queue: dict[str, float]) -> tuple[list[str], dict[str, int]]:
        """Phase order for one cycle, and how many times each is served.

        The order is FIXED, not sorted by demand. Reordering phases from cycle
        to cycle is how a driver ends up facing a green they did not expect,
        and it breaks the platoon progression the neighbouring junctions were
        timed against. Only the DURATION adapts.

        The one exception is re-service: under severe imbalance an arm can be
        given two shorter greens per cycle instead of one long one, which
        halves the wait for the vehicles that arrive just after it goes red.
        It costs one extra intergreen -- typically five seconds of pure lost
        time per cycle -- so it has to be asked for.
        """
        names = [p.name for p in self.phases]
        services = {n: 1 for n in names}
        if not self.allow_reservice or len(names) < 2:
            return names, services

        busiest = max(names, key=lambda n: queue.get(n, 0.0))
        others = [queue.get(n, 0.0) for n in names if n != busiest]
        rival = max(others) if others else 0.0
        if queue.get(busiest, 0.0) < self.reservice_ratio * max(rival, 1e-6):
            return names, services
        if queue.get(busiest, 0.0) <= 0:
            return names, services

        # A, B, A, C ... -- the busy phase comes round again after the first
        # other phase. Capped at two services so the cycle stays legible.
        services[busiest] = 2
        order: list[str] = []
        inserted = False
        for n in names:
            order.append(n)
            if n != busiest and not inserted:
                order.append(busiest)
                inserted = True
        return order, services

    def _limit_for(self, phase: str, services: dict[str, int]) -> tuple[float, float]:
        n = max(1, services.get(phase, 1))
        lo = self.min_green.get(phase, self.min_green_default)
        # min and max are per SERVICE -- a phase served twice needs twice the
        # total, and each of its two greens must still clear the crossing.
        return lo * n, self.max_green_s * n

    def _apply_limits(self, greens: dict[str, float], cycle: float, lost: float,
                      services: dict[str, int]) -> tuple[dict[str, float], float, list[str]]:
        """Hold every phase inside its safety limits, then rebalance.

        Water-filling: pin whatever hits a limit, redistribute what is left
        among the phases still free to move, repeat. Where the minimums simply
        do not fit inside the cycle ceiling, the minimums win and the cycle
        stretches -- a pedestrian's crossing time is not negotiable against a
        tidy cycle length.
        """
        notes: list[str] = []
        effective = cycle - lost
        pinned: dict[str, float] = {}
        free = dict(greens)

        for _ in range(len(greens) + 2):
            moved = False
            for name, val in list(free.items()):
                lo, hi = self._limit_for(name, services)
                if val < lo - 1e-6:
                    pinned[name] = lo
                    free.pop(name)
                    moved = True
                elif val > hi + 1e-6:
                    pinned[name] = hi
                    free.pop(name)
                    notes.append(f'{name} is demand-capped at {hi:.0f}s to stop '
                                 f'it starving the other arms')
                    moved = True
            if not moved:
                break
            remaining = effective - sum(pinned.values())
            if not free:
                break
            if remaining <= 0:
                # The cycle cannot fund what is left. Everyone still free drops
                # to their minimum and the cycle stretches to fit -- flagged
                # below, because it means the junction is past its capacity.
                for name in list(free):
                    pinned[name] = self._limit_for(name, services)[0]
                free = {}
                break
            total = sum(free.values())
            free = ({k: v / total * remaining for k, v in free.items()} if total > 0
                    else {k: remaining / len(free) for k in free})

        # Rebuild in PHASE order. Water-filling pins phases in whatever order
        # they hit a limit, and a plan that prints its phases out of order is
        # read as a plan that runs them out of order.
        merged = {**pinned, **free}
        out = {p.name: merged[p.name] for p in self.phases if p.name in merged}
        out.update({k: v for k, v in merged.items() if k not in out})
        new_cycle = sum(out.values()) + lost
        if new_cycle > self.max_cycle_s + 1e-6:
            notes.append(f'minimum greens force a {new_cycle:.0f}s cycle, past '
                         f'the {self.max_cycle_s:.0f}s ceiling -- this junction '
                         f'has more phases than its capacity supports')
        return out, new_cycle, notes

    def _rate_limit(self, greens: dict[str, float], caveats: list[str]) -> dict[str, float]:
        """No phase moves more than `max_delta_s` from the previous plan.

        Drivers learn a junction's rhythm within a few cycles. A plan that
        lurches produces exactly the late-amber running that red-light
        enforcement exists to punish, so the controller converges over several
        cycles instead of snapping.
        """
        prev = self._previous
        if prev is None or prev.mode == 'fixed':
            return greens
        out = {}
        for name, val in greens.items():
            before = prev.greens.get(name)
            if before is None:
                out[name] = val
                continue
            delta = val - before
            if abs(delta) > self.max_delta_s:
                out[name] = before + math.copysign(self.max_delta_s, delta)
                caveats.append(f'{name} is converging on its target '
                               f'({val:.0f}s) at {self.max_delta_s:.0f}s per cycle')
            else:
                out[name] = val
        return out

    @staticmethod
    def _imbalance(mode: str, y: dict[str, float], queue: dict[str, float],
                   greens: dict[str, float], critical: dict[str, str]) -> str:
        """The sentence a traffic engineer actually wants to read.

        It names the phase that WON the split and the quantity that decided it,
        because those are two different things depending on the mode. Under
        Webster the driver is the flow RATIO, and equal queues on unequal arms
        are not equal demand -- a single-lane arm carrying the same flow as a
        two-lane arm has twice the ratio and has genuinely earned more green.
        Once the junction is over capacity the objective changes to draining
        the longest queue, so the sentence changes with it.
        """
        if not greens:
            return ''
        top = max(greens, key=lambda k: greens[k])
        low = min(greens, key=lambda k: greens[k])
        arm = critical.get(top) or top

        if abs(greens[top] - greens[low]) < 1.0:
            return 'Demand is even across the phases; the split is held level.'

        gap = f'{greens[top]:.0f}s of green against {greens[low]:.0f}s for {low}'
        total_q = sum(queue.values())

        if mode == 'oversaturated':
            share = 100.0 * queue.get(top, 0.0) / total_q if total_q > 0 else 0.0
            return (f'{top} is holding {share:.0f}% of the standing queue '
                    f'(worst arm: {arm}), so it takes {gap}.')

        qs = (f' It is also carrying '
              f'{100.0 * queue.get(top, 0.0) / total_q:.0f}% of the standing '
              f'queue.' if total_q > 0 else '')
        return (f'{top} has the highest flow ratio (y={y.get(top, 0.0):.2f} on its '
                f'critical arm {arm}), so it takes {gap}.{qs}')

    def _materially_different(self, plan: SignalPlan) -> bool:
        prev = self._previous
        if prev is None or prev.mode != plan.mode or prev.order != plan.order:
            return True
        return any(abs(plan.greens.get(k, 0.0) - v) >= 1.0
                   for k, v in prev.greens.items())
