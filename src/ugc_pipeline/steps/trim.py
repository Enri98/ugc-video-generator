"""Trim step for the UGC pipeline.

Implements SPEC.md §5 Step 7 — end_of_clip_trim.

For each raw Veo clip, trims the last N seconds (default 0.5 s) to remove
the visual freeze/fade artifact that Veo 3.1 Fast typically appends.
Output is one trimmed MP4 per clip, stored alongside the raw clips in the
per-video artifacts directory.

The step is idempotent: if the trimmed artifact already exists on disk with
a non-zero size, the ffmpeg call is skipped.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import structlog

from ugc_pipeline.models import VideoState
from ugc_pipeline.utils.ffmpeg import FfmpegError, ffprobe_duration_seconds, run_ffmpeg

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-clip entry point
# ---------------------------------------------------------------------------


async def run_trim_for_clip(
    video_state: VideoState,
    clip_index: int,
    *,
    artifacts_root: Path,
    trim_tail_seconds: float = 0.5,
) -> Path:
    """Trim the tail from one raw clip and return the trimmed file path.

    Parameters
    ----------
    video_state:
        Mutable VideoState for the current video; ``artifacts`` is updated in
        place when the trimmed file is written.
    clip_index:
        0-based index of the clip to trim.
    artifacts_root:
        Root directory under which per-video artifact subdirectories live.
    trim_tail_seconds:
        Duration (in seconds) to remove from the end of the raw clip.
        Default is 0.5 s per SPEC.md §5 Step 7.

    Returns
    -------
    Path
        Absolute path to the trimmed MP4 file.

    Raises
    ------
    FfmpegError
        If ``trim_tail_seconds`` is greater than or equal to the clip
        duration, or if the ffmpeg subprocess exits non-zero.
    """
    t_start = time.monotonic()

    raw_key = f"clip_{clip_index}_raw"
    trimmed_key = f"clip_{clip_index}_trimmed"

    # ------------------------------------------------------------------
    # Idempotency check
    # ------------------------------------------------------------------
    existing_str = video_state.artifacts.get(trimmed_key)
    if existing_str is not None:
        existing = Path(existing_str)
        if existing.exists() and existing.stat().st_size > 0:
            log.info(
                "step_skipped_idempotent",
                step="end_of_clip_trim",
                video_id=video_state.video_id,
                clip_index=clip_index,
            )
            return existing

    log.info(
        "step_started",
        step="end_of_clip_trim",
        video_id=video_state.video_id,
        clip_index=clip_index,
    )

    # ------------------------------------------------------------------
    # Resolve raw artifact
    # ------------------------------------------------------------------
    raw_path = Path(video_state.artifacts[raw_key])

    # ------------------------------------------------------------------
    # Probe duration and compute target
    # ------------------------------------------------------------------
    duration = ffprobe_duration_seconds(raw_path)
    target = duration - trim_tail_seconds
    if target <= 0:
        raise FfmpegError(
            f"trim_tail_seconds={trim_tail_seconds} >= clip duration={duration} "
            f"for {raw_path!r}. Reduce trim_tail_seconds or inspect the raw clip."
        )

    # ------------------------------------------------------------------
    # Build output path and run ffmpeg
    # ------------------------------------------------------------------
    out_path = artifacts_root / video_state.video_id / f"clip_{clip_index}_trimmed.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "-y",
        "-i", str(raw_path),
        "-t", f"{target:.3f}",
        "-c", "copy",
        str(out_path),
    ]
    await run_ffmpeg(args)

    # ------------------------------------------------------------------
    # Update artifacts and log completion
    # ------------------------------------------------------------------
    video_state.artifacts[trimmed_key] = str(out_path)

    duration_ms = int((time.monotonic() - t_start) * 1000)
    log.info(
        "step_completed",
        step="end_of_clip_trim",
        video_id=video_state.video_id,
        clip_index=clip_index,
        artifact_path=str(out_path),
        duration_ms=duration_ms,
    )

    return out_path


# ---------------------------------------------------------------------------
# All-clips entry point
# ---------------------------------------------------------------------------


def _raw_clip_indices(video_state: VideoState) -> list[int]:
    """Return sorted list of clip indices that have a ``clip_{i}_raw`` artifact."""
    pattern = re.compile(r"^clip_(\d+)_raw$")
    indices: list[int] = []
    for key in video_state.artifacts:
        m = pattern.match(key)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


async def run_trim_all(
    video_state: VideoState,
    *,
    artifacts_root: Path,
    trim_tail_seconds: float = 0.5,
) -> list[Path]:
    """Trim all raw clips for a video and return the list of trimmed paths.

    Iterates over every ``clip_{i}_raw`` artifact key in *video_state*, in
    ascending order of *i*, and calls :func:`run_trim_for_clip` for each.

    Parameters
    ----------
    video_state:
        Mutable VideoState; artifacts dict updated in place.
    artifacts_root:
        Root directory under which per-video artifact subdirectories live.
    trim_tail_seconds:
        Tail duration to trim from each clip (default 0.5 s).

    Returns
    -------
    list[Path]
        Ordered list of trimmed MP4 file paths (one per clip).
    """
    indices = _raw_clip_indices(video_state)
    results: list[Path] = []
    for idx in indices:
        trimmed_path = await run_trim_for_clip(
            video_state,
            idx,
            artifacts_root=artifacts_root,
            trim_tail_seconds=trim_tail_seconds,
        )
        results.append(trimmed_path)
    return results
