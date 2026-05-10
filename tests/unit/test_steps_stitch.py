"""Unit tests for src/ugc_pipeline/steps/stitch.py.

Uses the ``clip_fixture_mp4_path`` session fixture (8-second 1080×1920 MP4)
for real ffmpeg round-trip tests.  No paid API calls are made.
"""

from __future__ import annotations

import pathlib
import shutil
import uuid

import pytest

from ugc_pipeline.models import CostBreakdown, VideoState
from ugc_pipeline.steps.stitch import cleanup_after_step, run_stitch
from ugc_pipeline.utils.ffmpeg import ffprobe_duration_seconds


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


def _setup_two_trimmed_clips(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
    video_state: VideoState,
) -> None:
    """Copy the fixture as clip_0_trimmed and clip_1_trimmed for *video_state*."""
    video_dir = tmp_path / video_state.video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    for i in range(2):
        dst = video_dir / f"clip_{i}_trimmed.mp4"
        shutil.copy2(clip_fixture_mp4_path, dst)
        video_state.artifacts[f"clip_{i}_trimmed"] = str(dst)


# ---------------------------------------------------------------------------
# Happy path: stitch two clips
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_stitch_happy_path(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Stitching two 8 s clips should produce a ~16 s stitched.mp4."""
    video_state = _make_video_state()
    _setup_two_trimmed_clips(clip_fixture_mp4_path, tmp_path, video_state)

    out_path = await run_stitch(
        video_state,
        artifacts_root=tmp_path,
        cleanup_raw=False,
        cleanup_trimmed=False,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0

    duration = ffprobe_duration_seconds(out_path)
    assert abs(duration - 16.0) < 0.5, f"Unexpected stitched duration: {duration}"

    assert video_state.artifacts.get("stitched") == str(out_path)


# ---------------------------------------------------------------------------
# Regression: relative artifacts_root must not produce a doubled path
# in the concat list (ffmpeg's concat demuxer resolves entries against
# the concat file's directory).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_stitch_relative_artifacts_root(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When artifacts_root is relative, concat entries must still be absolute."""
    monkeypatch.chdir(tmp_path)
    artifacts_root = pathlib.Path("artifacts")  # relative, like the real CLI

    video_state = _make_video_state()
    video_dir = artifacts_root / video_state.video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        dst = video_dir / f"clip_{i}_trimmed.mp4"
        shutil.copy2(clip_fixture_mp4_path, dst)
        video_state.artifacts[f"clip_{i}_trimmed"] = str(dst)  # relative path

    out_path = await run_stitch(
        video_state,
        artifacts_root=artifacts_root,
        cleanup_raw=False,
        cleanup_trimmed=False,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Concat list entries must be absolute, not relative.
    concat_text = (video_dir / "concat_list.txt").read_text(encoding="utf-8")
    for line in concat_text.splitlines():
        # Format: file '<path>'
        path_str = line.removeprefix("file ").strip().strip("'")
        assert pathlib.Path(path_str).is_absolute(), (
            f"Concat entry must be absolute, got: {path_str!r}"
        )


# ---------------------------------------------------------------------------
# Cleanup: raw and trimmed deleted after stitch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_stitch_cleanup_deletes_trimmed(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """cleanup_trimmed=True should remove trimmed files from disk and artifacts."""
    video_state = _make_video_state()
    _setup_two_trimmed_clips(clip_fixture_mp4_path, tmp_path, video_state)

    # Also add fake raw artifacts pointing to the same fixture so we can
    # verify raw cleanup independently
    video_dir = tmp_path / video_state.video_id
    for i in range(2):
        raw_dst = video_dir / f"clip_{i}_raw.mp4"
        shutil.copy2(clip_fixture_mp4_path, raw_dst)
        video_state.artifacts[f"clip_{i}_raw"] = str(raw_dst)

    await run_stitch(
        video_state,
        artifacts_root=tmp_path,
        cleanup_raw=True,
        cleanup_trimmed=True,
    )

    # Trimmed files should be deleted
    assert not any(k.endswith("_trimmed") for k in video_state.artifacts)
    # Raw files should be deleted
    assert not any(k.endswith("_raw") for k in video_state.artifacts)


# ---------------------------------------------------------------------------
# Idempotency: second call skips ffmpeg
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_stitch_idempotent(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Second call with existing stitched artifact should skip re-processing."""
    video_state = _make_video_state()
    _setup_two_trimmed_clips(clip_fixture_mp4_path, tmp_path, video_state)

    out_path = await run_stitch(
        video_state,
        artifacts_root=tmp_path,
        cleanup_raw=False,
        cleanup_trimmed=False,
    )
    first_mtime = out_path.stat().st_mtime

    # Remove trimmed artifacts so a re-run would fail if it tried to use them
    for i in range(2):
        trimmed_path = pathlib.Path(video_state.artifacts.get(f"clip_{i}_trimmed", ""))
        if trimmed_path.exists():
            trimmed_path.unlink()

    out_path2 = await run_stitch(
        video_state,
        artifacts_root=tmp_path,
        cleanup_raw=False,
        cleanup_trimmed=False,
    )
    assert out_path2 == out_path
    assert out_path2.stat().st_mtime == first_mtime


# ---------------------------------------------------------------------------
# cleanup_after_step helper (unit test, no ffmpeg)
# ---------------------------------------------------------------------------


def test_cleanup_after_step_removes_raw_and_trimmed(tmp_path: pathlib.Path) -> None:
    """cleanup_after_step with both flags False should delete both artifact types."""
    video_state = _make_video_state()

    # Create dummy files and register them as artifacts
    for i in range(2):
        for suffix in ("_raw", "_trimmed"):
            f = tmp_path / f"clip_{i}{suffix}.mp4"
            f.write_bytes(b"dummy")
            video_state.artifacts[f"clip_{i}{suffix}"] = str(f)

    cleanup_after_step(
        video_state,
        "stitch",
        {"keep_raw_clips": False, "keep_trimmed_clips": False},
    )

    assert not any(k.endswith("_raw") for k in video_state.artifacts)
    assert not any(k.endswith("_trimmed") for k in video_state.artifacts)
    # Files deleted from disk
    for i in range(2):
        for suffix in ("_raw", "_trimmed"):
            assert not (tmp_path / f"clip_{i}{suffix}.mp4").exists()


def test_cleanup_after_step_keep_raw(tmp_path: pathlib.Path) -> None:
    """keep_raw_clips=True should preserve raw artifacts."""
    video_state = _make_video_state()

    raw = tmp_path / "clip_0_raw.mp4"
    raw.write_bytes(b"dummy")
    video_state.artifacts["clip_0_raw"] = str(raw)

    cleanup_after_step(
        video_state,
        "stitch",
        {"keep_raw_clips": True, "keep_trimmed_clips": False},
    )

    assert "clip_0_raw" in video_state.artifacts
    assert raw.exists()


def test_cleanup_after_step_noop_for_other_steps(tmp_path: pathlib.Path) -> None:
    """cleanup_after_step should do nothing for steps other than 'stitch'."""
    video_state = _make_video_state()

    dummy = tmp_path / "clip_0_raw.mp4"
    dummy.write_bytes(b"dummy")
    video_state.artifacts["clip_0_raw"] = str(dummy)

    cleanup_after_step(video_state, "caption", {"keep_raw_clips": False})

    assert "clip_0_raw" in video_state.artifacts
    assert dummy.exists()
