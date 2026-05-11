"""Caption step for the UGC pipeline.

Workflow:
1. Run faster-whisper against the stitched MP4 audio track to produce a
   word-level transcript.
2. Render an .ass subtitle file with configurable font, size, and line wrap.
3. Burn the subtitles into the video with ffmpeg (-vf subtitles=...).
4. Write the final captioned MP4.

The step is idempotent: if ``final`` artifact exists with a non-zero size,
the step is skipped.

Language note: pipeline structure and code are English; the voiceover script
blocks are Italian, so faster-whisper is configured with ``language="it"``.
"""

from __future__ import annotations

import asyncio
import textwrap
import time
from pathlib import Path
from typing import Any

import structlog

from ugc_pipeline.models import VideoSpec, VideoState
from ugc_pipeline.utils.ffmpeg import FfmpegError, run_ffmpeg

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level WhisperModel cache
# ---------------------------------------------------------------------------

_whisper_model_cache: Any = None  # type: ignore[type-arg]
_whisper_model_key: tuple[str, str, str] | None = None  # (model_size, device, compute_type)


# ---------------------------------------------------------------------------
# Transcription helper
# ---------------------------------------------------------------------------


def transcribe_with_whisper(
    audio_or_video_path: Path,
    *,
    model_size: str = "base",
    language: str = "it",
) -> list[dict]:
    """Transcribe *audio_or_video_path* using faster-whisper.

    Loads the WhisperModel once at module level and reuses it for subsequent
    calls with the same configuration, avoiding costly reload overhead.

    Parameters
    ----------
    audio_or_video_path:
        Path to an audio or video file. faster-whisper uses libsndfile /
        ffmpeg under the hood to decode audio from video containers.
    model_size:
        Model size identifier (e.g. ``"base"``, ``"small"``, ``"medium"``).
    language:
        BCP-47 language code. Defaults to ``"it"`` (Italian) per pipeline
        spec.

    Returns
    -------
    list[dict]
        One dict per segment::

            {
                "start": float,   # segment start in seconds
                "end":   float,   # segment end in seconds
                "text":  str,     # Italian transcription (UTF-8)
                "words": [{"start": float, "end": float, "word": str}, ...],
            }
    """
    global _whisper_model_cache, _whisper_model_key

    cache_key = (model_size, "cpu", "int8")
    if _whisper_model_cache is None or _whisper_model_key != cache_key:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        _whisper_model_cache = WhisperModel(
            model_size, device="cpu", compute_type="int8"
        )
        _whisper_model_key = cache_key

    model = _whisper_model_cache
    segments_iter, _ = model.transcribe(
        str(audio_or_video_path),
        language=language,
        word_timestamps=True,
    )

    result: list[dict] = []
    for seg in segments_iter:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({"start": w.start, "end": w.end, "word": w.word})
        result.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": words,
            }
        )
    return result


# ---------------------------------------------------------------------------
# .ass renderer
# ---------------------------------------------------------------------------


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert *seconds* to SubStation Alpha time format ``H:MM:SS.cc``."""
    centiseconds = int(round(seconds * 100))
    cs = centiseconds % 100
    total_s = centiseconds // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _wrap_text(text: str, max_chars: int) -> str:
    """Hard-wrap *text* at *max_chars* characters using ``\\N`` (ASS line break).

    textwrap.wrap handles Italian non-ASCII transparently because it operates
    on Unicode codepoints.
    """
    lines = textwrap.wrap(text.strip(), width=max_chars, break_long_words=True)
    return r"\N".join(lines)


def render_ass(
    segments: list[dict],
    *,
    max_chars_per_line: int = 30,
    video_w: int = 720,
    video_h: int = 1280,
    font: str = "Arial",
    font_size: int = 44,
) -> str:
    """Render a SubStation Alpha v4+ (.ass) subtitle string.

    Produces a fully-formed .ass document with:
    - ``[Script Info]`` section with ``PlayResX`` / ``PlayResY``.
    - ``[V4+ Styles]`` section with one Default style (white text, black
      outline, bottom-centre alignment).
    - ``[Events]`` section with one ``Dialogue`` line per *segment*.

    Italian non-ASCII characters (è, à, ò, …) are preserved literally —
    no escaping or transliteration.

    Parameters
    ----------
    segments:
        List of segment dicts as returned by :func:`transcribe_with_whisper`.
        An empty list produces a header-only .ass file with no Dialogue lines.
    max_chars_per_line:
        Maximum characters per subtitle line before a hard wrap is inserted.
    video_w, video_h:
        Pixel dimensions for ``PlayResX`` / ``PlayResY``.
    font:
        Font family name for the subtitle style.
    font_size:
        Font size in pixels.

    Returns
    -------
    str
        UTF-8 string containing the complete .ass file content.
    """
    # Outline thickness and shadow depth
    outline = 2
    shadow = 0

    # Alignment: 2 = bottom-centre in ASS numpad alignment
    alignment = 2

    # Colour format: &HAABBGGRR (alpha, blue, green, red) — white with no alpha
    primary_colour = "&H00FFFFFF"   # white
    outline_colour = "&H00000000"   # black
    back_colour = "&H00000000"

    header = (
        "[Script Info]\n"
        f"PlayResX: {video_w}\n"
        f"PlayResY: {video_h}\n"
        "ScriptType: v4.00+\n"
        "Collisions: Normal\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{font_size},{primary_colour},&H000000FF,"
        f"{outline_colour},{back_colour},0,0,0,0,100,100,0,0,1,"
        f"{outline},{shadow},{alignment},10,10,20,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    dialogue_lines: list[str] = []
    for seg in segments:
        start = _seconds_to_ass_time(seg["start"])
        end = _seconds_to_ass_time(seg["end"])
        text = _wrap_text(seg.get("text", ""), max_chars_per_line)
        dialogue_lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
        )

    return header + "\n".join(dialogue_lines) + ("\n" if dialogue_lines else "")


# ---------------------------------------------------------------------------
# Windows-safe ffmpeg subtitles filter path
# ---------------------------------------------------------------------------


def _ass_filter_path(p: Path) -> str:
    """Return the .ass file path formatted for ffmpeg's subtitles filter.

    On Windows, backslashes must be converted to forward slashes and colons
    (from drive letters, e.g. ``C:``) must be escaped as ``\\:``. Single
    quotes inside the path are escaped as ``\\\\'``.

    Parameters
    ----------
    p:
        Absolute path to the .ass file.

    Returns
    -------
    str
        A string suitable for embedding in ``subtitles='...'``.
    """
    s = str(p).replace("\\", "/")
    # Escape colon (drive letter separator on Windows: C:/ → C\:/)
    s = s.replace(":", "\\:")
    # Escape single quotes
    s = s.replace("'", "\\\\'")
    return s


# ---------------------------------------------------------------------------
# Main step entry point
# ---------------------------------------------------------------------------


async def run_caption(
    video_state: VideoState,
    spec: VideoSpec,
    *,
    artifacts_root: Path,
    max_chars_per_line: int = 30,
    video_w: int = 720,
    video_h: int = 1280,
    font: str = "Arial",
    font_size: int = 44,
    whisper_model_size: str = "base",
) -> Path:
    """Transcribe, render subtitles, and burn captions into the stitched video.

    Parameters
    ----------
    video_state:
        Mutable VideoState; ``artifacts["captions_ass"]`` and
        ``artifacts["final"]`` are set on success.
    spec:
        VideoSpec for the current video (used for metadata; not directly
        needed for transcription in v1 — audio drives the transcript).
    artifacts_root:
        Root directory under which per-video artifact subdirectories live.
    max_chars_per_line:
        Maximum characters per subtitle line before wrapping.

    Returns
    -------
    Path
        Absolute path to the final captioned MP4 file.

    Raises
    ------
    FfmpegError
        If ffmpeg exits with a non-zero return code during burn-in.
    KeyError
        If the ``"stitched"`` artifact is not present in *video_state*.
    """
    t_start = time.monotonic()

    # ------------------------------------------------------------------
    # Idempotency check
    # ------------------------------------------------------------------
    existing_str = video_state.artifacts.get("final")
    if existing_str is not None:
        existing = Path(existing_str)
        if existing.exists() and existing.stat().st_size > 0:
            log.info(
                "step_skipped_idempotent",
                step="caption",
                video_id=video_state.video_id,
            )
            return existing

    log.info(
        "step_started",
        step="caption",
        video_id=video_state.video_id,
    )

    video_dir = artifacts_root / video_state.video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    stitched_path = Path(video_state.artifacts["stitched"])

    # ------------------------------------------------------------------
    # Transcription (run in thread pool to avoid blocking the event loop)
    # ------------------------------------------------------------------
    segments = await asyncio.to_thread(
        transcribe_with_whisper, stitched_path, model_size=whisper_model_size
    )

    if not segments:
        log.warning(
            "caption_empty_transcript",
            video_id=video_state.video_id,
            stitched_path=str(stitched_path),
        )

    # ------------------------------------------------------------------
    # Render .ass subtitle file
    # ------------------------------------------------------------------
    ass_text = render_ass(
        segments,
        max_chars_per_line=max_chars_per_line,
        video_w=video_w,
        video_h=video_h,
        font=font,
        font_size=font_size,
    )
    ass_path = video_dir / "captions.ass"
    ass_path.write_text(ass_text, encoding="utf-8")
    video_state.artifacts["captions_ass"] = str(ass_path.resolve())

    # ------------------------------------------------------------------
    # Burn subtitles into video with ffmpeg
    # ------------------------------------------------------------------
    out_path = video_dir / "final.mp4"
    filter_path = _ass_filter_path(ass_path)
    vf_arg = f"subtitles='{filter_path}'"

    args = [
        "-y",
        "-i", str(stitched_path),
        "-vf", vf_arg,
        "-c:a", "copy",
        str(out_path),
    ]
    await run_ffmpeg(args)

    # ------------------------------------------------------------------
    # Update artifacts and log completion
    # ------------------------------------------------------------------
    video_state.artifacts["final"] = str(out_path.resolve())

    duration_ms = int((time.monotonic() - t_start) * 1000)
    log.info(
        "step_completed",
        step="caption",
        video_id=video_state.video_id,
        artifact_path=str(out_path),
        duration_ms=duration_ms,
    )

    return out_path
