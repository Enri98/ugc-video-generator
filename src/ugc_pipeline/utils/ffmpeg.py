"""ffmpeg utilities for the UGC pipeline.

Provides helpers to locate the ffmpeg binary (via imageio_ffmpeg),
probe media file properties (via PyAV), run async ffmpeg subprocesses,
and format concat-list paths.
"""

from __future__ import annotations

import asyncio
import re
from asyncio.subprocess import PIPE
from pathlib import Path

# ---------------------------------------------------------------------------
# Module-level cache for the ffmpeg binary path
# ---------------------------------------------------------------------------

_ffmpeg_binary_cache: str | None = None


def ffmpeg_binary() -> str:
    """Return the path to the ffmpeg binary bundled with imageio_ffmpeg.

    The result is cached at module level so that the imageio_ffmpeg lookup
    runs at most once per process.
    """
    global _ffmpeg_binary_cache
    if _ffmpeg_binary_cache is None:
        import imageio_ffmpeg  # type: ignore[import-untyped]

        _ffmpeg_binary_cache = imageio_ffmpeg.get_ffmpeg_exe()
    return _ffmpeg_binary_cache


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class FfmpegError(Exception):
    """Raised when an ffmpeg subprocess exits with a non-zero return code."""


# ---------------------------------------------------------------------------
# PyAV-based probing helpers
# ---------------------------------------------------------------------------


def ffprobe_duration_seconds(path: Path) -> float:
    """Return the duration of the media file at *path* in seconds.

    Uses PyAV (which bundles its own libav) rather than relying on a system
    ffprobe binary.

    Parameters
    ----------
    path:
        Absolute or relative path to any media file readable by libav.

    Returns
    -------
    float
        Duration in seconds.

    Raises
    ------
    FfmpegError
        If the file cannot be opened or has no duration information.
    """
    try:
        import av  # type: ignore[import-untyped]

        container = av.open(str(path))
        try:
            if container.duration is None:
                raise FfmpegError(
                    f"No duration information found in {path!r}."
                )
            duration = container.duration / av.time_base
        finally:
            container.close()
    except FfmpegError:
        raise
    except Exception as exc:
        raise FfmpegError(
            f"Failed to probe duration of {path!r}: {exc}"
        ) from exc

    return float(duration)


def ffprobe_has_audio(path: Path) -> bool:
    """Return True if the file at *path* contains at least one audio stream.

    Uses PyAV (which bundles its own libav) rather than relying on a system
    ffprobe binary.

    Parameters
    ----------
    path:
        Absolute or relative path to any media file readable by libav.

    Returns
    -------
    bool
        True if the container has at least one audio stream, False otherwise.

    Raises
    ------
    FfmpegError
        If the file cannot be opened.
    """
    try:
        import av  # type: ignore[import-untyped]

        container = av.open(str(path))
        try:
            has_audio = any(s.type == "audio" for s in container.streams)
        finally:
            container.close()
    except FfmpegError:
        raise
    except Exception as exc:
        raise FfmpegError(
            f"Failed to probe audio streams of {path!r}: {exc}"
        ) from exc

    return has_audio


def ffprobe_dimensions(path: Path) -> tuple[int, int]:
    """Return the (width, height) of the first video stream in *path*.

    Uses PyAV for probing — no system ffprobe required.

    Parameters
    ----------
    path:
        Absolute or relative path to any media file readable by libav.

    Returns
    -------
    tuple[int, int]
        ``(width, height)`` in pixels.

    Raises
    ------
    FfmpegError
        If the file cannot be opened or contains no video stream.
    """
    try:
        import av  # type: ignore[import-untyped]

        container = av.open(str(path))
        try:
            video_stream = next(
                (s for s in container.streams if s.type == "video"),
                None,
            )
            if video_stream is None:
                raise FfmpegError(
                    f"No video stream found in {path!r}."
                )
            width = video_stream.width
            height = video_stream.height
        finally:
            container.close()
    except FfmpegError:
        raise
    except Exception as exc:
        raise FfmpegError(
            f"Failed to probe dimensions of {path!r}: {exc}"
        ) from exc

    if not width or not height:
        raise FfmpegError(
            f"Video stream in {path!r} has zero dimensions: {width}x{height}."
        )

    return (width, height)


# ---------------------------------------------------------------------------
# Async ffmpeg runner
# ---------------------------------------------------------------------------


async def run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg with the given argument list.

    The binary path is resolved via :func:`ffmpeg_binary` (imageio_ffmpeg).
    stdout and stderr are captured. On non-zero exit, :class:`FfmpegError`
    is raised with the captured stderr included in the message.

    Parameters
    ----------
    args:
        Argument list (NOT including the ffmpeg binary itself).

    Raises
    ------
    FfmpegError
        If ffmpeg exits with a non-zero return code.
    """
    binary = ffmpeg_binary()
    proc = await asyncio.create_subprocess_exec(
        binary,
        *args,
        stdout=PIPE,
        stderr=PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        stderr_text = stderr.decode(errors="replace")
        raise FfmpegError(
            f"ffmpeg exited with code {proc.returncode}.\n"
            f"Command: {binary} {' '.join(args)}\n"
            f"Stderr:\n{stderr_text}"
        )


# ---------------------------------------------------------------------------
# Concat-list path quoting
# ---------------------------------------------------------------------------


def quote_concat_path(path: Path) -> str:
    """Return the path string formatted for an ffmpeg concat list file.

    The ffmpeg concat demuxer expects paths to be wrapped in single quotes.
    Any existing single quotes in the path are escaped as ``'\\''``.

    Parameters
    ----------
    path:
        Path to a media file referenced in the concat list.

    Returns
    -------
    str
        The quoted path string suitable for use as the value after
        ``file `` in a concat list (i.e. WITHOUT the ``file `` prefix).
        Example: ``'path/to/clip_0_trimmed.mp4'``
    """
    path_str = str(path)
    # Escape any existing single quotes
    escaped = path_str.replace("'", "'\\''")
    return f"'{escaped}'"
