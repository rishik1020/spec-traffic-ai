"""
senti.evidence.encode
=====================
Make a written video playable in a browser.

THE PROBLEM
OpenCV writes MP4 with the `mp4v` tag -- MPEG-4 Part 2. No browser decodes it.
Chrome fails with `DEMUXER_ERROR_NO_SUPPORTED_STREAMS`, which looks exactly
like a loading spinner that never finishes, so the evidence clips in the
officer portal appeared to be "still buffering" when in fact nothing was ever
going to play. Every clip the system had ever written was unviewable in the one
place it needed to be viewed.

Asking OpenCV for H.264 instead does not work here either: its encoder needs
`openh264-2.5.0-win64.dll`, which is absent, and it silently falls back to
producing a file that opens without error and contains nothing usable. A codec
that fails loudly would have been easier to find than one that fails this way.

THE FIX
ffmpeg is on PATH, so the file is transcoded after writing: H.264 baseline,
yuv420p, and `+faststart` so the moov atom sits at the front and the video
begins playing before it has fully downloaded.

WHY TRANSCODE RATHER THAN PIPE FRAMES TO FFMPEG
Piping would avoid the second pass, but it makes every writer in the codebase
depend on ffmpeg being present and on a subprocess staying alive for the length
of a run. Transcoding afterwards keeps ffmpeg optional: when it is missing the
original file survives, the caller is told once, and everything except
in-browser playback still works.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_WARNED = False


def ffmpeg_available() -> bool:
    return shutil.which('ffmpeg') is not None


def to_browser_mp4(path: str | Path, crf: int = 23,
                   quiet: bool = True) -> Path:
    """Re-encode `path` in place as browser-playable H.264.

    Returns the path either way. A failure here is never fatal: the original
    file is left alone and still opens in VLC, a desktop player, or ffmpeg --
    it just will not play in the portal.
    """
    global _WARNED
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return p

    if not ffmpeg_available():
        if not _WARNED:
            print('[senti] ffmpeg not found -- clips stay in mp4v and will NOT '
                  'play in a browser. Install ffmpeg to view them in the portal.')
            _WARNED = True
        return p

    tmp = p.with_suffix('.h264.mp4')
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', str(p),
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(crf),
        # yuv420p and the even-dimension filter between them cover the two
        # things that make an H.264 file refuse to play: an unsupported pixel
        # format, and a width or height that is odd.
        '-pix_fmt', 'yuv420p',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-movflags', '+faststart',
        '-an', str(tmp),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        if not quiet:
            print(f'[senti] transcode failed ({e}) -- keeping {p.name}')
        tmp.unlink(missing_ok=True)
        return p

    if r.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        if not quiet:
            print(f'[senti] transcode failed -- keeping {p.name}\n'
                  f'{(r.stderr or "")[:300]}')
        tmp.unlink(missing_ok=True)
        return p

    p.unlink(missing_ok=True)
    tmp.rename(p)
    return p
