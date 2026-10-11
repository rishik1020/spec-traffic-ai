"""
senti.engine
============
Wires the pieces together.

    frame -> detect -> track -> rules -> rolling buffer -> evidence package

The engine owns no violation logic. Every rule is loaded from the camera's YAML
profile, which is what makes "bike camera vs junction camera vs highway camera"
a CONFIGURATION difference rather than three codebases -- one shared perception
pass, different rules enabled.

The same perception pass also feeds `senti.signal`, which measures congestion
per junction arm and recommends signal timings. Detection is expensive and runs
ONCE; enforcement and control are two readers of the same FrameResult. That is
the whole argument for a shared perception layer -- adding adaptive control
added no per-frame inference cost at all.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .calibration import Calibration
from .core.types import Detection, FrameResult
from .evidence.buffer import RollingBuffer
from .evidence.package import EvidenceWriter
from .ingest.source import VideoSource
from .perception.detector import TrafficDetector, read_signal_state
# importing the package autoloads and registers every rule module
from .rules import Rule, Violation, available_rules, get_rule
from .signal import AdaptiveController, DemandMeter, SignalPlan




class Engine:
    def __init__(
        self,
        config_path: str | Path,
        weights: Optional[str] = None,
        evidence_root: str | Path = 'data/evidence',
        show: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
        self.camera_id = self.config.get('camera_id', 'cam')
        self.show = show

        # Per-camera geometry. Built ONCE and handed to every rule via context,
        # so lane lookups are not recomputed per rule per frame.
        self.calibration = Calibration.from_dict(self.config.get('calibration'))

        self.detector = TrafficDetector(
            weights=weights or self.config.get('weights',
                                               'runs/driveindia/weights/best.pt'),
            imgsz=int(self.config.get('imgsz', 640)),
            device=device,
        )

        # only the rules this camera enables -- see the class docstring
        self.rules: list[Rule] = []
        for name, rule_cfg in (self.config.get('rules') or {}).items():
            if rule_cfg is False:
                continue
            cfg = rule_cfg if isinstance(rule_cfg, dict) else {}
            self.rules.append(get_rule(name)(cfg))

        if not self.rules:
            print(f'[senti] WARNING: no rules enabled. available: {available_rules()}')

        # Adaptive signal timing. Enabled only when the camera profile has a
        # `signal_control:` block AND the calibration names its lanes' arms --
        # a highway camera has neither, so the feature simply is not there.
        sig_cfg = self.config.get('signal_control') or {}
        self.demand: Optional[DemandMeter] = None
        self.controller: Optional[AdaptiveController] = None
        if sig_cfg and sig_cfg is not False:
            ctrl = AdaptiveController(sig_cfg)
            if not self.calibration.approach_names:
                print('[senti] signal_control is configured but no lane declares '
                      'an `approach:` -- congestion cannot be attributed to an '
                      'arm, so adaptive timing is OFF.')
            elif not ctrl.is_configured:
                print('[senti] signal_control needs at least two phases -- '
                      'adaptive timing is OFF.')
            else:
                self.demand = DemandMeter(self.calibration, sig_cfg)
                self.controller = ctrl
        self.plans: list[SignalPlan] = []
        self.plan_log = Path(evidence_root).parent / 'signal' / f'{self.camera_id}.jsonl'

        self.writer = EvidenceWriter(Path(evidence_root), self.camera_id)
        self.buffer: Optional[RollingBuffer] = None

        self._track_frames: dict[int, int] = {}
        # (x, y, t) per track, for the on-screen speed readout only.
        # Rules keep their own history; this exists so the preview can show
        # speed without reaching into a rule's internals.
        self._track_pos: dict[int, list] = {}
        self._pending: dict[str, dict] = {}
        self.violations: list[Violation] = []
        # track id -> (keep boxed until this frame, rule name)
        self._flash: dict[int, tuple] = {}
        self.events: list[dict] = []

    # ---------------------------------------------------------------------

    def run(self, source: str | int, max_frames: Optional[int] = None,
            annotate: Optional[str | Path] = None) -> list[Violation]:
        src = VideoSource(source, stride=int(self.config.get('stride', 1)),
                          max_frames=max_frames)
        info = src.open()

        ev = self.config.get('evidence', {}) or {}
        self.buffer = RollingBuffer(
            fps=info.fps,
            pre_seconds=float(ev.get('pre_seconds', 5.0)),
            post_seconds=float(ev.get('post_seconds', 3.0)),
            downscale=ev.get('downscale'),
        )

        print(f'[senti] camera={self.camera_id}  {info.width}x{info.height} '
              f'@ {info.fps:.1f}fps  live={info.is_live}')
        print(f'[senti] rules: {[r.name for r in self.rules]}')
        cal = self.calibration
        if cal.is_calibrated:
            print(f'[senti] calibration: {len(cal.lanes)} lane(s), '
                  f'{len(cal.stop_lines)} stop line(s), '
                  f'homography {"yes" if cal.homography else "no"}')
        else:
            print('[senti] calibration: NONE -- geometric rules will abstain.')
            print('[senti]   run scripts/calibrate.py to draw lanes for this camera.')
        print(f'[senti] model: {"DriveIndia" if self.detector.indian_model else "COCO fallback"}')
        if self.controller is not None:
            print(f'[senti] adaptive signal: {len(self.controller.phases)} phase(s) '
                  f'over arms {cal.approach_names}, recomputed every '
                  f'{self.controller.update_every_s:.0f}s -- ADVISORY ONLY')

        writer = None
        if annotate:
            import cv2
            Path(annotate).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(annotate),
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     info.fps or 25.0, (info.width, info.height))
            print(f'[senti] annotating -> {annotate}')

        for idx, ts, frame in src.frames():
            result = self.detector.track(frame, frame_index=idx, timestamp=ts)
            self.buffer.push(idx, ts, frame)

            for d in result.detections:
                if d.track_id is not None:
                    self._track_frames[d.track_id] = self._track_frames.get(d.track_id, 0) + 1
                    hist = self._track_pos.setdefault(d.track_id, [])
                    bx, by = d.bottom_center
                    hist.append((bx, by, ts))
                    if len(hist) > 30:
                        hist.pop(0)

            if self.demand is not None:
                self.demand.update(result, ts)
                self._signal_tick(ts)

            context = self._build_context(frame, result)

            for rule in self.rules:
                for v in rule.process(result, context):
                    self._on_violation(v, rule, frame, result, context)

            self._drain(frame)

            if writer is not None:
                writer.write(self._draw(frame, result, context))
            if self.show:
                self._preview(frame, result, context)

            if idx % 100 == 0 and idx:
                print(f'  frame {idx:6d} | {len(result.detections):3d} objs | '
                      f'PCU {result.total_pcu:6.1f} | {result.inference_ms:5.1f}ms | '
                      f'buf {self.buffer.memory_mb:5.0f}MB | '
                      f'violations {len(self.violations)}')

        # end of stream: anything still collecting post-roll gets a short clip
        for key, frames in self.buffer.flush_all().items():
            self._finalise(key, frames, None)

        if writer is not None:
            writer.release()
            from .evidence.encode import to_browser_mp4
            to_browser_mp4(annotate)
            self._write_events(Path(annotate))

        if self.show:
            import cv2
            cv2.destroyAllWindows()

        print(f'\n[senti] done. {len(self.violations)} violation(s) -> {self.writer.root}')
        if self.plans:
            print(f'[senti] {len(self.plans)} signal recommendation(s) -> {self.plan_log}')
            print(f'[senti] last: {self.plans[-1].summary()}')
        return self.violations

    # ---------------------------------------------------------------------

    def _write_events(self, video: Path) -> None:
        """Timeline for the monitor screen, written beside the video.

        The monitor needs to know WHEN each violation happened so clicking one
        can seek the player to it. Reading that back out of the evidence
        folders would work but couples the monitor to the evidence layout;
        a timeline written next to the video it describes does not.
        """
        import json as _json
        out = video.with_suffix('.events.json')
        out.write_text(_json.dumps({
            'camera_id': self.camera_id,
            'video': video.name,
            'events': self.events,
        }, indent=2), encoding='utf-8')
        print(f'[senti] {len(self.events)} event(s) -> {out}')

    def _signal_tick(self, ts: float) -> None:
        """Recompute the recommended timing plan, if one is due.

        Written to a JSONL log rather than acted on. The log IS the deliverable:
        it lets a traffic engineer replay every recommendation against the
        demand that produced it, and it is the input a SUMO simulation needs.
        Nothing here touches a signal head -- see senti/signal/controller.py for
        why that boundary is deliberate.
        """
        plan = self.controller.update(self.demand.read(), ts,
                                      wall_clock=datetime.now())
        if plan is None:
            return
        self.plans.append(plan)

        self.plan_log.parent.mkdir(parents=True, exist_ok=True)
        with self.plan_log.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(plan.to_dict()) + '\n')

        if plan.changed:
            print(f'  [~] signal @{ts:6.1f}s  {plan.summary()}')
            print(f'      {plan.reason}')
            for c in plan.caveats:
                print(f'      ! {c}')

    # ---------------------------------------------------------------------

    def _build_context(self, frame, result: FrameResult) -> dict:
        """Shared per-frame context every rule can read.

        Computed ONCE here rather than per rule -- reading the signal in three
        separate rules would triple the work and could yield three answers.
        """
        ctx: dict = {'signal_state': 'unknown', 'signal_score': 0.0,
                     'signal_source': 'none',
                     'total_pcu': result.total_pcu,
                     'calibration': self.calibration}

        head = self.calibration.signal_head
        if head is not None:
            # A drawn signal head beats a detected one. The lamp is bolted to a
            # pole, so its position is known once and forever, and relying on
            # the detector to re-find it every frame fails exactly when it
            # matters -- at night, in rain, at distance.
            x1, y1, x2, y2 = head
            box = Detection(cls_name='traffic_light', confidence=1.0,
                            xyxy=(float(x1), float(y1), float(x2), float(y2)))
            state, score = read_signal_state(frame, box)
            ctx.update(signal_state=state, signal_score=score,
                       signal_source='calibrated')
        elif result.traffic_lights:
            # Fall back to the biggest detected head. On a junction camera
            # seeing several arms this can be the WRONG signal -- draw the head
            # instead whenever a rule depends on it.
            biggest = max(result.traffic_lights, key=lambda d: d.area)
            state, score = read_signal_state(frame, biggest)
            ctx.update(signal_state=state, signal_score=score,
                       signal_source='detected')
        return ctx

    def _on_violation(self, v: Violation, rule: Rule, frame, result, context) -> None:
        key = f'{v.rule_name}_t{v.track_id}_f{v.frame_index}'
        if self.buffer.is_capturing(key):
            return
        self.buffer.start_capture(key)
        self._pending[key] = {
            'violation': v,
            'rule': rule,
            'subject': result.by_track(v.track_id),
            'violation_frame': frame.copy(),
            'context': dict(context),
            'track_frames': self._track_frames.get(v.track_id, 0),
        }
        self.violations.append(v)
        self._flash[v.track_id] = (v.frame_index + 60, v.rule_name)
        self.events.append({
            'evidence_id': f'{self.camera_id}_{v.key}',
            'rule': v.rule_name, 'track_id': v.track_id,
            'vehicle_class': v.cls_name, 'frame_index': v.frame_index,
            'seconds': round(v.timestamp, 2), 'reason': v.reason,
            'confidence': round(v.confidence, 3),
        })
        print(f'  [!] {v.rule_name} track#{v.track_id} frame {v.frame_index}: {v.reason}')

    def _drain(self, frame) -> None:
        for key in self.buffer.ready():
            self._finalise(key, self.buffer.pop(key), frame)

    def _finalise(self, key: str, frames: list, frame) -> None:
        meta = self._pending.pop(key, None)
        if meta is None:
            return
        approach = frames[0].frame if frames else None
        folder = self.writer.write(
            violation=meta['violation'],
            clip_frames=frames,
            buffer=self.buffer,
            subject=meta['subject'],
            violation_frame=meta['violation_frame'],
            approach_frame=approach,
            plate_conf=0.0,                       # ANPR not wired yet
            signal_state=meta['context'].get('signal_state', 'unknown'),
            signal_score=meta['context'].get('signal_score', 0.0),
            track_frames=meta['track_frames'],
            rule=meta['rule'],
        )
        print(f'      -> evidence written: {folder.name}')

    # ---------------------------------------------------------------------

    def _speed_kmph(self, track_id, window: int = 12):
        """Display-only speed. Returns None when it cannot be measured."""
        hom = getattr(self.calibration, 'homography', None)
        pts = self._track_pos.get(track_id, [])
        if hom is None or len(pts) < 6:
            return None
        recent = pts[-window:]
        dt = recent[-1][2] - recent[0][2]
        if dt <= 1e-3:
            return None
        try:
            m = hom.distance_m((recent[0][0], recent[0][1]),
                               (recent[-1][0], recent[-1][1]))
        except Exception:
            return None
        v = (m / dt) * 3.6
        return v if 0.0 <= v <= 250.0 else None

    def _label(self, d) -> str:
        base = f'{d.cls_name} #{d.track_id}'
        v = self._speed_kmph(d.track_id) if d.track_id is not None else None
        return f'{base} {v:.0f} km/h' if v is not None else f'{base} {d.confidence:.2f}'

    def _preview(self, frame, result: FrameResult, context: dict) -> None:
        import cv2
        cv2.imshow('SPEC Traffic AI', self._draw(frame, result, context))
        cv2.waitKey(1)

    def _draw(self, frame, result: FrameResult, context: dict):
        """Annotate one frame. Used by the live preview AND the video export.

        Shared on purpose: an exported video that does not look exactly like
        what the operator saw is a different artefact, and the one a reviewer
        is shown later had better be the one the system actually produced.
        """
        import cv2
        import numpy as np
        import supervision as sv

        if not result.detections:
            canvas = frame
        else:
            dets = sv.Detections(
                xyxy=np.array([d.xyxy for d in result.detections], dtype=float),
                confidence=np.array([d.confidence for d in result.detections]),
                class_id=np.zeros(len(result.detections), dtype=int),
                tracker_id=np.array([d.track_id if d.track_id is not None else -1
                                     for d in result.detections]),
            )
            labels = [self._label(d) for d in result.detections]
            canvas = sv.BoxAnnotator().annotate(frame.copy(), dets)
            canvas = sv.LabelAnnotator(text_scale=0.4).annotate(canvas, dets, labels)

        cv2.putText(canvas,
                    f'f{result.frame_index} | PCU {result.total_pcu:.1f} | '
                    f'signal {context.get("signal_state")} | '
                    f'violations {len(self.violations)}',
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.demand is not None:
            # Per-arm queue, drawn on the arm itself. Seeing the number sitting
            # on the lanes it was measured from is the fastest way to catch a
            # badly drawn polygon -- a busy arm reading zero is a geometry bug,
            # not a quiet road.
            y = 52
            for arm, d in self.demand.read().items():
                cv2.putText(canvas,
                            f'{arm}: queue {d.queue_pcu:5.1f} PCU '
                            f'({d.queue_count}) | flow {d.flow_pcu_hr:5.0f} PCU/hr'
                            + ('  [TRUNCATED]' if d.truncated else ''),
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 200, 255) if d.truncated else (0, 255, 180), 1)
                y += 20
            if self.plans:
                cv2.putText(canvas, self.plans[-1].summary(), (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1)

        # A violation that already fired on this track stays boxed for a
        # couple of seconds. The instant itself is one frame out of thirty --
        # anyone watching at normal speed would miss it entirely.
        for tid, (until, rule) in list(self._flash.items()):
            if result.frame_index > until:
                self._flash.pop(tid, None)
                continue
            d = result.by_track(tid)
            if d is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in d.xyxy)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 160, 240), 3)
            tag = f'{rule.replace("_", " ").upper()} #{tid}'
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 8)),
                          (x1 + tw + 8, y1), (60, 160, 240), -1)
            cv2.putText(canvas, tag, (x1 + 4, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2)

        return canvas
