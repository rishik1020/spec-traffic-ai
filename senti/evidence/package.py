"""
senti.evidence.package
======================
Writes one self-contained folder per violation, and scores its own evidence.

    data/evidence/hyd01_wrong_way_t47_f3942/
    |-- clip.mp4               rolling-buffer window: approach -> event -> exit
    |-- frame_violation.jpg    the moment the rule fired (full resolution)
    |-- frame_approach.jpg     context, ~2s earlier
    |-- evidence.json          everything the portal and challan need

EVIDENCE DEFENSIBILITY SCORE (EDS)
The novel part. Detection confidence answers "how sure am I this is a
motorcycle". EDS answers a different and, for enforcement, far more important
question: "would this challan survive being contested?"

Those come apart constantly. A crisp 0.97-confidence detection of a vehicle
whose plate is unreadable and which was 60% occluded at the moment of violation
is a weak challan. EDS measures that, names the weak dimension, and routes
accordingly -- so an officer spends their time on cases worth reviewing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cv2

from ..core.types import Detection
from ..rules.base import Violation

IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class Defensibility:
    """Per-dimension evidence quality, 0-1 each, plus the weighted total."""

    plate: float = 0.0
    visibility: float = 0.0
    track_integrity: float = 0.0
    rule_margin: float = 0.0
    context: float = 0.0
    score: float = 0.0            # 0-100
    verdict: str = 'review'       # auto | review | drop
    weakest: str = ''
    # Which dimensions actually contributed. Plate recognition is not built
    # yet, and scoring a system on a capability it does not have is not
    # humility, it is a broken instrument -- see `compute`.
    dimensions_used: tuple = ()
    plate_available: bool = False

    WEIGHTS = {
        'plate': 0.30,            # no plate, no challan -- weighted highest
        'visibility': 0.20,
        'track_integrity': 0.20,
        'rule_margin': 0.15,
        'context': 0.15,
    }

    @classmethod
    def compute(
        cls,
        violation: Violation,
        subject: Optional[Detection],
        frame_shape: tuple[int, int],
        plate_conf: float = 0.0,
        signal_score: float = 0.0,
        track_frames: int = 0,
        plate_available: bool = False,
    ) -> 'Defensibility':
        """Score the evidence over the dimensions that can actually be measured.

        WHY `plate_available` EXISTS
        Plate carries the heaviest weight, 0.30, because a challan without an
        identified vehicle is not a challan. But ANPR is not implemented, so
        plate_conf is always 0 -- which silently capped every score ever
        produced at 70/100 and made the `auto` band at 80 mathematically
        unreachable. Every package in the system was being judged against a
        ceiling it could not touch, and the three-band triage had quietly
        become a two-band one.

        The fix is not to invent a plate score. It is to score only what was
        measured and say so: when plate is unavailable its weight is
        redistributed across the remaining dimensions in proportion, and
        `dimensions_used` records exactly which contributed. The number stays
        comparable across packages and the omission stays visible, which is the
        same bargain the rest of the system makes when it abstains.
        """
        d = cls()
        d.plate_available = bool(plate_available)

        # 1. plate -- can the vehicle actually be identified?
        d.plate = max(0.0, min(1.0, plate_conf))

        # 2. visibility -- how much of the frame does the subject occupy?
        # A vehicle 15px across cannot support a contested fine regardless of
        # how confident the detector was.
        if subject is not None:
            fh, fw = frame_shape[:2]
            rel = subject.area / float(fw * fh)
            d.visibility = max(0.0, min(1.0, rel / 0.02))   # 2% of frame = full marks

        # 3. track integrity -- was this the same vehicle throughout?
        # Short tracks mean the tracker only just acquired it, so attribution
        # to the vehicle that actually committed the act is shaky.
        d.track_integrity = max(0.0, min(1.0, track_frames / 30.0))

        # 4. rule margin -- how unambiguous was the breach?
        # Sustained frames beyond the minimum is the generic proxy; rules that
        # have a natural margin (px past a line, degrees off-heading) override
        # it via detail['margin_norm'].
        margin = violation.detail.get('margin_norm')
        if margin is None:
            sustained = violation.detail.get('sustained_frames', 0)
            margin = min(1.0, sustained / 20.0)
        d.rule_margin = max(0.0, min(1.0, float(margin)))

        # 5. context -- was the surrounding situation certain?
        # For signal rules this is the HSV pixel fraction. Rules with no context
        # dependency get a neutral pass rather than being punished for it.
        d.context = max(0.0, min(1.0, signal_score / 0.15)) if signal_score > 0 else 0.6

        parts = {
            'plate': d.plate, 'visibility': d.visibility,
            'track_integrity': d.track_integrity, 'rule_margin': d.rule_margin,
            'context': d.context,
        }
        if not plate_available:
            parts.pop('plate')
        d.dimensions_used = tuple(parts)

        # Renormalise so the weights of whatever remains still sum to 1. The
        # score therefore always spans the full 0-100 range and the bands keep
        # their meaning, whichever dimensions were available.
        total_w = sum(cls.WEIGHTS[k] for k in parts) or 1.0
        d.score = round(100.0 * sum(parts[k] * cls.WEIGHTS[k] for k in parts)
                        / total_w, 1)
        d.weakest = min(parts, key=parts.get)
        d.verdict = 'auto' if d.score >= 80 else ('review' if d.score >= 50 else 'drop')
        return d

    def explain(self) -> str:
        base = (f'EDS {self.score:.0f}/100 ({self.verdict}); weakest dimension: '
                f'{self.weakest} ({getattr(self, self.weakest):.2f})')
        if not self.plate_available:
            base += '; scored WITHOUT plate (ANPR not wired)'
        return base


class EvidenceWriter:
    def __init__(self, root: Path, camera_id: str) -> None:
        self.root = Path(root)
        self.camera_id = camera_id
        self.root.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        violation: Violation,
        clip_frames: list,
        buffer,
        subject: Optional[Detection] = None,
        violation_frame=None,
        approach_frame=None,
        plate_conf: float = 0.0,
        signal_state: str = 'unknown',
        signal_score: float = 0.0,
        track_frames: int = 0,
        rule=None,
    ) -> Path:
        folder = self.root / f'{self.camera_id}_{violation.key}'
        folder.mkdir(parents=True, exist_ok=True)

        clip_path = None
        if clip_frames:
            clip_path = buffer.write_clip(clip_frames, folder / 'clip.mp4')

        shape = (1080, 1920)
        if violation_frame is not None:
            cv2.imwrite(str(folder / 'frame_violation.jpg'), violation_frame)
            shape = violation_frame.shape
        if approach_frame is not None:
            cv2.imwrite(str(folder / 'frame_approach.jpg'), approach_frame)

        eds = Defensibility.compute(
            violation, subject, shape,
            plate_conf=plate_conf, signal_score=signal_score,
            track_frames=track_frames,
        )

        record = {
            'evidence_id': f'{self.camera_id}_{violation.key}',
            'camera_id': self.camera_id,
            'violation': {
                'rule': violation.rule_name,
                'description': getattr(rule, 'description', ''),
                'mv_act_section': getattr(rule, 'mv_act_section', ''),
                'track_id': violation.track_id,
                'vehicle_class': violation.cls_name,
                'detection_confidence': round(violation.confidence, 4),
                'bbox_xyxy': [round(v, 1) for v in violation.xyxy],
            },
            # Frame dimensions travel WITH the box. A bounding box in pixels
            # is meaningless to anything that does not already know how big the
            # frame was, and the portal draws that box over an image scaled to
            # whatever width the browser gave it.
            'frame_size': [int(shape[1]), int(shape[0])],
            'timing': {
                'frame_index': violation.frame_index,
                'stream_seconds': round(violation.timestamp, 3),
                'recorded_at_ist': datetime.now(IST).strftime('%d-%m-%Y %H:%M:%S IST'),
            },
            'context': {
                'signal_state': signal_state,
                'signal_confidence': round(signal_score, 4),
                'track_frames': track_frames,
            },
            # the plain-language line an officer reads before approving
            'reason_trace': violation.reason,
            'detail': violation.detail,
            'defensibility': asdict(eds),
            'artifacts': {
                'clip': clip_path.name if clip_path else None,
                'violation_frame': 'frame_violation.jpg' if violation_frame is not None else None,
                'approach_frame': 'frame_approach.jpg' if approach_frame is not None else None,
            },
            'review': {'status': 'pending', 'officer': None, 'decision_at': None,
                       'plate_number': None, 'rejection_reason': None},
        }

        (folder / 'evidence.json').write_text(
            json.dumps(record, indent=2), encoding='utf-8')
        return folder
