"""
senti.engine
============
Wires the pieces together.

    frame -> detect -> track -> rules -> rolling buffer -> evidence package

The engine owns no violation logic. Every rule is loaded from the camera's YAML
profile, which is what makes "bike camera vs junction camera vs highway camera"
a CONFIGURATION difference rather than three codebases -- one shared perception
pass, different rules enabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from .core.types import FrameResult
from .evidence.buffer import RollingBuffer
from .evidence.package import EvidenceWriter
from .ingest.source import VideoSource
from .perception.detector import TrafficDetector, read_signal_state
# importing the package autoloads and registers every rule module
from .rules import Rule, Violation, available_rules, get_rule




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

        self.writer = EvidenceWriter(Path(evidence_root), self.camera_id)
        self.buffer: Optional[RollingBuffer] = None

        self._track_frames: dict[int, int] = {}
        self._pending: dict[str, dict] = {}
        self.violations: list[Violation] = []

    # ---------------------------------------------------------------------

    def run(self, source: str | int, max_frames: Optional[int] = None) -> list[Violation]:
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
        print(f'[senti] model: {"DriveIndia" if self.detector.indian_model else "COCO fallback"}')

        for idx, ts, frame in src.frames():
            result = self.detector.track(frame, frame_index=idx, timestamp=ts)
            self.buffer.push(idx, ts, frame)

            for d in result.detections:
                if d.track_id is not None:
                    self._track_frames[d.track_id] = self._track_frames.get(d.track_id, 0) + 1

            context = self._build_context(frame, result)

            for rule in self.rules:
                for v in rule.process(result, context):
                    self._on_violation(v, rule, frame, result, context)

            self._drain(frame)

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

        if self.show:
            import cv2
            cv2.destroyAllWindows()

        print(f'\n[senti] done. {len(self.violations)} violation(s) -> {self.writer.root}')
        return self.violations

    # ---------------------------------------------------------------------

    def _build_context(self, frame, result: FrameResult) -> dict:
        """Shared per-frame context every rule can read.

        Computed ONCE here rather than per rule -- reading the signal in three
        separate rules would triple the work and could yield three answers.
        """
        ctx: dict = {'signal_state': 'unknown', 'signal_score': 0.0,
                     'total_pcu': result.total_pcu}
        lights = result.traffic_lights
        if lights:
            biggest = max(lights, key=lambda d: d.area)
            state, score = read_signal_state(frame, biggest)
            ctx['signal_state'] = state
            ctx['signal_score'] = score
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

    def _preview(self, frame, result: FrameResult, context: dict) -> None:
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
            labels = [f'{d.cls_name} #{d.track_id} {d.confidence:.2f}'
                      for d in result.detections]
            canvas = sv.BoxAnnotator().annotate(frame.copy(), dets)
            canvas = sv.LabelAnnotator(text_scale=0.4).annotate(canvas, dets, labels)

        cv2.putText(canvas,
                    f'f{result.frame_index} | PCU {result.total_pcu:.1f} | '
                    f'signal {context.get("signal_state")} | '
                    f'violations {len(self.violations)}',
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('SPEC Traffic AI', canvas)
        cv2.waitKey(1)
