"""Unit tests for src/ugc_pipeline/steps/trim.py.

Uses the ``clip_fixture_mp4_path`` session fixture (8-second 1080×1920 MP4)
for real ffmpeg-based round-trip tests.  No paid API calls are made.
"""

from __future__ import annotations

import pathlib
import shutil
import uuid

import pytest

from ugc_pipeline.models import VideoState, CostBreakdown
from ugc_pipeline.steps.trim import run_trim_all, run_trim_for_clip
from ugc_pipeline.utils.ffmpeg import FfmpegError, ffprobe_duration_seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_video_state(video_id: str | None = None) -> VideoState:
    vid = video_id or str(uuid.uuid4())
    return VideoState(
        video_id=vid,
        product_id="test_product",
        spec_index=0,
        costs_usd=CostBreakdown(),
    )


# ---------------------------------------------------------------------------
# Happy path: single clip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_trim_for_clip_happy_path(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Trimming the 8 s fixture by 0.5 s should produce a ~7.5 s output."""
    video_state = _make_video_state()
    video_dir = tmp_path / video_state.video_id
    video_dir.mkdir(parents=True)

    # Copy fixture into per-video dir as raw clip
    raw_path = video_dir / "clip_0_raw.mp4"
    shutil.copy2(clip_fixture_mp4_path, raw_path)
    video_state.artifacts["clip_0_raw"] = str(raw_path)

    out_path = await run_trim_for_clip(
        video_state,
        0,
        artifacts_root=tmp_path,
        trim_tail_seconds=0.5,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0

    duration = ffprobe_duration_seconds(out_path)
    assert abs(duration - 7.5) < 0.2, f"Unexpected duration: {duration}"

    # Artifact key should be updated
    assert video_state.artifacts.get("clip_0_trimmed") == str(out_path)


# ---------------------------------------------------------------------------
# Error: trim tail >= duration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_trim_for_clip_raises_if_tail_too_large(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Trim tail >= clip duration should raise FfmpegError."""
    video_state = _make_video_state()
    video_dir = tmp_path / video_state.video_id
    video_dir.mkdir(parents=True)

    raw_path = video_dir / "clip_0_raw.mp4"
    shutil.copy2(clip_fixture_mp4_path, raw_path)
    video_state.artifacts["clip_0_raw"] = str(raw_path)

    with pytest.raises(FfmpegError, match="trim_tail_seconds"):
        await run_trim_for_clip(
            video_state,
            0,
            artifacts_root=tmp_path,
            trim_tail_seconds=100.0,  # way larger than 8 s fixture
        )


# ---------------------------------------------------------------------------
# Idempotency: second call skips re-processing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_trim_for_clip_idempotent(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Second call with existing trimmed artifact should skip ffmpeg."""
    video_state = _make_video_state()
    video_dir = tmp_path / video_state.video_id
    video_dir.mkdir(parents=True)

    raw_path = video_dir / "clip_0_raw.mp4"
    shutil.copy2(clip_fixture_mp4_path, raw_path)
    video_state.artifacts["clip_0_raw"] = str(raw_path)

    # First call: produces trimmed file
    out_path = await run_trim_for_clip(
        video_state, 0, artifacts_root=tmp_path, trim_tail_seconds=0.5
    )
    first_mtime = out_path.stat().st_mtime

    # Remove the raw artifact to confirm second call does NOT re-run ffmpeg
    # (if it tried to re-read the raw path after deletion it would fail)
    raw_path.unlink()

    # Second call: should return the cached trimmed path unchanged
    out_path2 = await run_trim_for_clip(
        video_state, 0, artifacts_root=tmp_path, trim_tail_seconds=0.5
    )
    assert out_path2 == out_path
    assert out_path2.stat().st_mtime == first_mtime


# ---------------------------------------------------------------------------
# run_trim_all: multiple clips
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_trim_all_two_clips(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """run_trim_all should trim both raw clips in order."""
    video_state = _make_video_state()
    video_dir = tmp_path / video_state.video_id
    video_dir.mkdir(parents=True)

    for i in range(2):
        raw_path = video_dir / f"clip_{i}_raw.mp4"
        shutil.copy2(clip_fixture_mp4_path, raw_path)
        video_state.artifacts[f"clip_{i}_raw"] = str(raw_path)

    results = await run_trim_all(
        video_state, artifacts_root=tmp_path, trim_tail_seconds=0.5
    )

    assert len(results) == 2
    for path in results:
        assert path.exists()
        assert path.stat().st_size > 0
        dur = ffprobe_duration_seconds(path)
        assert abs(dur - 7.5) < 0.2
