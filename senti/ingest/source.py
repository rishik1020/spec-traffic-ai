"""
senti.ingest.source
===================
One interface in front of every way frames can arrive.

This is the file that makes "uploaded video now, live camera later" a config
change rather than a rewrite. Nothing downstream knows or cares whether the
frames came off disk, an RTSP stream, or a webcam.

    VideoSource('clip.mp4')                      # MVP
    VideoSource('rtsp://10.0.0.5:554/stream1')   # production
    VideoSource(0)                               # webcam
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional, Union

import cv2


def resolve_source(source):
    """Turn a convenient source string into something OpenCV can open.

    YouTube live traffic cameras are a good stand-in for municipal CCTV -- they
    are genuinely live, genuinely traffic, and published for public viewing. But
    a YouTube page URL is not a video stream; yt-dlp has to resolve it to an HLS
    manifest first, and those manifests are ~1500 chars and expire after a few
    hours. Resolving here means callers can just pass the watch URL.

    Anything that is not a YouTube link passes straight through untouched.
    """
    if not isinstance(source, str):
        return source
    if 'youtube.com' not in source and 'youtu.be' not in source:
        return source

    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        raise RuntimeError('YouTube source needs yt-dlp:  pip install yt-dlp')

    print('[senti] resolving YouTube stream via yt-dlp ...')
    opts = {'quiet': True, 'no_warnings': True, 'format': 'best[height<=720]/best'}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(source, download=False)

    url = info.get('url')
    if not url:
        fmts = [f for f in info.get('formats', []) if f.get('url')]
        if not fmts:
            raise RuntimeError('yt-dlp returned no playable stream')
        url = fmts[-1]['url']

    live = ' (LIVE)' if info.get('is_live') else ''
    # YouTube titles routinely contain emoji, and a Windows cp1252 console
    # raises UnicodeEncodeError trying to print them -- which would kill the
    # run before a single frame is read. Strip to ASCII for display only.
    title = str(info.get('title', '?'))[:70]
    title = title.encode('ascii', 'replace').decode('ascii')
    print(f'[senti] {title}{live}')
    return url


@dataclass
class StreamInfo:
    width: int
    height: int
    fps: float
    total_frames: int          # 0 for a live stream -- it has no end
    is_live: bool

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


class VideoSource:
    """Yields (frame_index, timestamp_seconds, frame) from any source."""

    def __init__(
        self,
        source: Union[str, int],
        stride: int = 1,
        max_frames: Optional[int] = None,
        reconnect_attempts: int = 3,
    ) -> None:
        """
        stride              process every Nth frame. 1 = every frame. Raising
                            this is the cheapest way to hit real-time budget on
                            a slow machine, at the cost of tracking continuity.
        max_frames          stop after N processed frames (testing).
        reconnect_attempts  live streams drop; files do not. Only applied when
                            the source looks live.
        """
        self.source = source
        self.stride = max(1, stride)
        self.max_frames = max_frames
        self.reconnect_attempts = reconnect_attempts

        self._cap: Optional[cv2.VideoCapture] = None
        self._info: Optional[StreamInfo] = None

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> StreamInfo:
        self.source = resolve_source(self.source)
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f'could not open video source: {self.source!r}')

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # A file reports a positive frame count; RTSP and webcams report 0.
        is_live = total <= 0 or isinstance(self.source, int)

        self._cap = cap
        self._info = StreamInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps if fps > 0 else 25.0,   # RTSP often lies; assume 25
            total_frames=max(0, total),
            is_live=is_live,
        )
        return self._info

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def info(self) -> StreamInfo:
        if self._info is None:
            return self.open()
        return self._info

    # -- iteration ---------------------------------------------------------

    def frames(self) -> Iterator[tuple[int, float, object]]:
        """Yield (frame_index, timestamp_seconds, frame)."""
        info = self.info
        cap = self._cap
        assert cap is not None

        raw_index = -1
        emitted = 0
        attempts = 0

        try:
            while True:
                ok, frame = cap.read()

                if not ok:
                    if not info.is_live or attempts >= self.reconnect_attempts:
                        break
                    # live stream hiccup -- reopen and keep going rather than
                    # ending the run on a momentary network drop
                    attempts += 1
                    cap.release()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(self.source)
                    self._cap = cap
                    continue

                attempts = 0
                raw_index += 1
                if raw_index % self.stride:
                    continue

                # Timestamp from the stream itself where available; falls back
                # to frame count / fps, which is right for files and close
                # enough for live.
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                ts = (ms / 1000.0) if ms and ms > 0 else (raw_index / info.fps)

                yield raw_index, ts, frame

                emitted += 1
                if self.max_frames is not None and emitted >= self.max_frames:
                    break
        finally:
            self.close()

    def __iter__(self):
        return self.frames()
