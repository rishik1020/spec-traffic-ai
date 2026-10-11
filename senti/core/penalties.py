"""
senti.core.penalties
====================
What a violation costs, and whether anyone has checked.

WHY A SCHEDULE AND NOT A NUMBER IN THE CODE
Penalty amounts are notified by the state and revised; the Motor Vehicles
(Amendment) Act 2019 sets central figures that states then compound their own
way. A number compiled into a rule is a number nobody will ever update. A YAML
file with a `verified` flag on every row is a number somebody can be asked to
check, and whose unchecked status travels with it.

WHY `verified` IS CARRIED ALL THE WAY TO THE SCREEN
The portal drafts a challan with an amount on it. If that amount is wrong the
document is worse than useless -- it is a wrong demand for money with a
statutory section printed beside it. So the flag is not advisory metadata; it
is rendered next to the figure, every time, until somebody sets it true against
the gazette.

WHY SOME OFFENCES BLOCK THE CHALLAN ENTIRELY
Over-speeding from vision is a screening signal: the rule already marks its own
evidence inadmissible because speed enforcement in India needs radar or LIDAR.
Lane discipline is advisory for a similar reason. Those rows carry
`blocks_challan: true`, and the portal refuses to let them be approved as a
fine no matter what the officer clicks. A system that can be clicked into
issuing an inadmissible challan does not have a human in the loop; it has a
human holding the bag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / 'config' / 'penalties.yaml'


@dataclass
class Penalty:
    rule: str
    section: str = ''
    offence: str = ''
    amount: Optional[int] = None
    amount_range: Optional[tuple] = None
    repeat_amount: Optional[int] = None
    licence_disqualification_months: Optional[int] = None
    verified: bool = False
    blocks_challan: bool = False
    notes: str = ''
    currency: str = 'INR'

    @property
    def display_amount(self) -> str:
        """What to print on the draft. A range stays a range."""
        if self.amount_range:
            lo, hi = self.amount_range
            return f'₹{lo:,} – ₹{hi:,}'
        if self.amount is None:
            return 'not set'
        return f'₹{self.amount:,}'

    @property
    def consequences(self) -> list[str]:
        out = []
        if self.licence_disqualification_months:
            out.append(f'Licence disqualification: '
                       f'{self.licence_disqualification_months} months')
        if self.repeat_amount:
            out.append(f'Repeat offence: ₹{self.repeat_amount:,}')
        return out


@dataclass
class Schedule:
    penalties: dict = field(default_factory=dict)
    currency: str = 'INR'
    schedule_verified: bool = False
    last_checked: Optional[str] = None

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> 'Schedule':
        p = Path(path)
        if not p.is_file():
            return cls()
        cfg = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
        cur = cfg.get('currency', 'INR')
        out = {}
        for rule, row in (cfg.get('penalties') or {}).items():
            row = row or {}
            rng = row.get('amount_range')
            out[rule] = Penalty(
                rule=rule,
                section=row.get('section', ''),
                offence=row.get('offence', ''),
                amount=row.get('amount'),
                amount_range=tuple(rng) if rng else None,
                repeat_amount=row.get('repeat_amount'),
                licence_disqualification_months=row.get(
                    'licence_disqualification_months'),
                verified=bool(row.get('verified', False)),
                blocks_challan=bool(row.get('blocks_challan', False)),
                notes=(row.get('notes') or '').strip(),
                currency=cur,
            )
        return cls(penalties=out, currency=cur,
                   schedule_verified=bool(cfg.get('schedule_verified', False)),
                   last_checked=cfg.get('last_checked'))

    def for_rule(self, rule: str) -> Penalty:
        """Always returns a Penalty. An unknown rule is unpriced, not free."""
        return self.penalties.get(rule) or Penalty(
            rule=rule, offence=rule.replace('_', ' ').title(),
            notes='No penalty row for this rule -- amount unknown.')
