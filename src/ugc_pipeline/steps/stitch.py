"""Stitch step for the UGC pipeline.

Implements SPEC.md §5 Step 8 — stitch.

Concatenates all trimmed clips for a video into a single stitched MP4 using
the ffmpeg concat demuxer. After a successful stitch, raw and trimmed clips
may be deleted per the cleanup policy flags passed to
:func:`cleanup_after_step`.

The step is idempotent: if the ``stitched`` artifact already exists on disk
with a non-zero size, the ffmpeg call is skipped.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import structlog

from ugc_pipeline.models import VideoState
from ugc_pipeline.utils.ffmpeg import FfmpegError, ffprobe_has_audio, quote_concat_path, run_ffmpeg

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _normalise_audio_streams(
    clip_paths: list[Path],
    video_dir: Path,
    *,
    video_state: VideoState,
    clip_indices: list[int],
) -> list[Path]:
    """Ensure all clips share the same audio-stream presence.

    Probes each clip. If all clips lack audio, returns the input list unchanged.
    If all clips have audio, returns the input list unchanged.
    If mixed: re-muxes each silent clip with a silent AAC audio track of matching
    duration, writing to ``video_dir/<original_stem>_silenced.mp4``. Returns
    a new list of paths (silenced clips swapped in for silent originals).
    Existing audio-bearing clips are passed through unchanged.

    Each silenced intermediate is registered in ``video_state.artifacts`` under
    the key ``clip_{i}_silenced`` so that :func:`cleanup_after_step` can delete
    it later.

    Parameters
    ----------
    clip_paths:
        Ordered list of trimmed clip paths.
    video_dir:
        Directory where silenced variants are written.
    video_state:
        Mutable VideoState; silenced artifact keys are registered inline.
    clip_indices:
        Clip indices in the same order as *clip_paths* (used to build the
        ``clip_{i}_silenced`` artifact key).

    Returns
    -------
    list[Path]
        Normalised list — same order as input.
    """
    has_audio_flags = [ffprobe_has_audio(p) for p in clip_paths]
    all_have_audio = all(has_audio_flags)
    none_have_audio = not any(has_audio_flags)

    if all_have_audio:
        log.info("stitch_audio_homogeneous", variant="all_have_audio", clip_count=len(clip_paths))
        return clip_paths

    if none_have_audio:
        log.info("stitch_audio_homogeneous", variant="none_have_audio", clip_count=len(clip_paths))
        return clip_paths

    # Mixed: add silent audio track to clips that lack one
    audio_clip_count = sum(has_audio_flags)
    silent_clip_count = len(clip_paths) - audio_clip_count
    log.warning(
        "stitch_audio_heterogeneous_normalised",
        silent_clip_count=silent_clip_count,
        audio_clip_count=audio_clip_count,
    )

    async def _silence_one(clip: Path, clip_index: int) -> Path:
        stem = clip.stem
        out = video_dir / f"{stem}_silenced.mp4"
        await run_ffmpeg([
            "-y",
            "-i", str(clip),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-shortest",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            str(out),
        ])
        video_state.artifacts[f"clip_{clip_index}_silenced"] = str(out.resolve())
        return out

    async def _maybe_silence(clip: Path, clip_index: int, needs_silence: bool) -> Path:
        if needs_silence:
            return await _silence_one(clip, clip_index)
        return clip

    results: list[Path] = await asyncio.gather(
        *[
            _maybe_silence(clip, clip_index, not has_audio)
            for clip, clip_index, has_audio in zip(clip_paths, clip_indices, has_audio_flags)
        ]
    )
    return list(results)


def _trimmed_clip_indices(video_state: VideoState) -> list[int]:
    """Return sorted list of clip indices that have a ``clip_{i}_trimmed`` artifact."""
    pattern = re.compile(r"^clip_(\d+)_trimmed$")
    indices: list[int] = []
    for key in video_state.artifacts:
        m = pattern.match(key)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


# ---------------------------------------------------------------------------
# cleanup_after_step hook (SPEC.md §7)
# ---------------------------------------------------------------------------


def cleanup_after_step(
    video_state: VideoState,
    step_name: str,
    cfg_dict: dict,
) -> None:
    """Delete per-step artifact files and remove their keys from *video_state*.

    Currently only acts on the ``"stitch"`` step, deleting raw and/or
    trimmed clip artifacts according to *cfg_dict*.

    Parameters
    ----------
    video_state:
        Mutable VideoState; artifact dict is mutated in place.
    step_name:
        Name of the step that just completed (e.g. ``"stitch"``).
    cfg_dict:
        Mapping of cleanup flags. Recognised keys:
        - ``"keep_raw_clips"`` (bool, default ``False``): if False, delete
          all ``clip_{i}_raw`` artifacts after stitch.
        - ``"keep_trimmed_clips"`` (bool, default ``False``): if False,
          delete all ``clip_{i}_trimmed`` artifacts after stitch.
    """
    if step_name != "stitch":
        return

    keep_raw = cfg_dict.get("keep_raw_clips", False)
    keep_trimmed = cfg_dict.get("keep_trimmed_clips", False)

    if not keep_raw:
        for key in [k for k in list(video_state.artifacts) if k.endswith("_raw")]:
            file_path = Path(video_state.artifacts[key])
            try:
                if file_path.exists():
                    file_path.unlink()
            except OSError as exc:
                log.warning(
                    "cleanup_delete_failed",
                    artifact_key=key,
                    path=str(file_path),
                    error=str(exc),
                )
            video_state.artifacts.pop(key, None)

    if not keep_trimmed:
        for key in [k for k in list(video_state.artifacts) if k.endswith("_trimmed")]:
            file_path = Path(video_state.artifacts[key])
            try:
                if file_path.exists():
                    file_path.unlink()
            except OSError as exc:
                log.warning(
                    "cleanup_delete_failed",
                    artifact_key=key,
                    path=str(file_path),
                    error=str(exc),
                )
            video_state.artifacts.pop(key, None)

        for key in [k for k in list(video_state.artifacts) if k.endswith("_silenced")]:
            file_path = Path(video_state.artifacts[key])
            try:
                if file_path.exists():
                    file_path.unlink()
            except OSError as exc:
                log.warning(
                    "cleanup_delete_failed",
                    artifact_key=key,
                    path=str(file_path),
                    error=str(exc),
                )
            video_state.artifacts.pop(key, None)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_stitch(
    video_state: VideoState,
    *,
    artifacts_root: Path,
    cleanup_raw: bool = True,
    cleanup_trimmed: bool = True,
) -> Path:
    """Concatenate trimmed clips into a single stitched MP4.

    Parameters
    ----------
    video_state:
        Mutable VideoState; ``artifacts["stitched"]`` is set on success.
    artifacts_root:
        Root directory under which per-video artifact subdirectories live.
    cleanup_raw:
        If True (default), delete ``clip_{i}_raw`` files and remove their
        artifact keys after a successful stitch.
    cleanup_trimmed:
        If True (default), delete ``clip_{i}_trimmed`` files and remove
        their artifact keys after a successful stitch.

    Returns
    -------
    Path
        Absolute path to the stitched MP4 file.

    Raises
    ------
    FfmpegError
        If ffmpeg exits with a non-zero return code.
    ValueError
        If there are no trimmed clips to stitch.
    """
    t_start = time.monotonic()

    # ------------------------------------------------------------------
    # Idempotency check
    # ------------------------------------------------------------------
    existing_str = video_state.artifacts.get("stitched")
    if existing_str is not None:
        existing = Path(existing_str)
        if existing.exists() and existing.stat().st_size > 0:
            log.info(
                "step_skipped_idempotent",
                step="stitch",
                video_id=video_state.video_id,
            )
            return existing

    log.info(
        "step_started",
        step="stitch",
        video_id=video_state.video_id,
    )

    # ------------------------------------------------------------------
    # Collect trimmed clips in order
    # ------------------------------------------------------------------
    indices = _trimmed_clip_indices(video_state)
    if not indices:
        raise ValueError(
            f"No trimmed clip artifacts found for video {video_state.video_id!r}. "
            "Run the trim step before stitch."
        )

    clip_paths = [
        Path(video_state.artifacts[f"clip_{i}_trimmed"]) for i in indices
    ]

    # ------------------------------------------------------------------
    # Normalise audio streams (handles mixed audio/no-audio clips)
    # ------------------------------------------------------------------
    video_dir = artifacts_root / video_state.video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = await _normalise_audio_streams(
        clip_paths, video_dir, video_state=video_state, clip_indices=indices
    )

    # ------------------------------------------------------------------
    # Write concat list
    # ------------------------------------------------------------------
    concat_list = video_dir / "concat_list.txt"

    # ffmpeg's concat demuxer resolves relative entries against the concat
    # file's directory, not the CWD — so we always write absolute paths to
    # avoid a doubled-prefix lookup when artifacts_root itself is relative.
    lines = [f"file {quote_concat_path(p.resolve())}" for p in clip_paths]
    concat_list.write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # Run ffmpeg concat
    # ------------------------------------------------------------------
    out_path = video_dir / "stitched.mp4"
    args = [
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(out_path),
    ]
    await run_ffmpeg(args)

    # ------------------------------------------------------------------
    # Update artifacts
    # ------------------------------------------------------------------
    video_state.artifacts["stitched"] = str(out_path.resolve())

    # ------------------------------------------------------------------
    # Cleanup (per SPEC.md §7 cleanup_after_step)
    # ------------------------------------------------------------------
    cfg_dict = {
        "keep_raw_clips": not cleanup_raw,
        "keep_trimmed_clips": not cleanup_trimmed,
    }
    cleanup_after_step(video_state, "stitch", cfg_dict)

    duration_ms = int((time.monotonic() - t_start) * 1000)
    log.info(
        "step_completed",
        step="stitch",
        video_id=video_state.video_id,
        artifact_path=str(out_path),
        duration_ms=duration_ms,
    )

    return out_path
