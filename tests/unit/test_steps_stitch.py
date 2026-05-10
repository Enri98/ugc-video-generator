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
from ugc_pipeline.steps.stitch import (
    _build_xfade_args,
    _normalise_audio_streams,
    cleanup_after_step,
    run_stitch,
)
from ugc_pipeline.utils.ffmpeg import ffprobe_duration_seconds, ffprobe_has_audio, run_ffmpeg


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


# ---------------------------------------------------------------------------
# Audio normalisation helpers
# ---------------------------------------------------------------------------


async def _add_silent_audio_track(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Add a synthetic silent AAC audio track to a video clip using ffmpeg."""
    await run_ffmpeg([
        "-y",
        "-i", str(src),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-shortest",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        str(dst),
    ])


# ---------------------------------------------------------------------------
# _normalise_audio_streams — direct unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalise_returns_input_unchanged_when_homogeneous_silent(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Two silent clips → input list returned as-is, no _silenced files."""
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    clip0 = video_dir / "clip_0_trimmed.mp4"
    clip1 = video_dir / "clip_1_trimmed.mp4"
    shutil.copy2(clip_fixture_mp4_path, clip0)
    shutil.copy2(clip_fixture_mp4_path, clip1)

    video_state = _make_video_state()
    result = await _normalise_audio_streams(
        [clip0, clip1], video_dir, video_state=video_state, clip_indices=[0, 1]
    )

    assert result == [clip0, clip1]
    assert not any(video_dir.glob("*_silenced.mp4"))
    assert not any(k.endswith("_silenced") for k in video_state.artifacts)


@pytest.mark.asyncio
async def test_normalise_returns_input_unchanged_when_homogeneous_audio(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Two clips that both have audio → input list returned as-is, no _silenced files."""
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    clip0 = video_dir / "clip_0_trimmed.mp4"
    clip1 = video_dir / "clip_1_trimmed.mp4"
    await _add_silent_audio_track(clip_fixture_mp4_path, clip0)
    await _add_silent_audio_track(clip_fixture_mp4_path, clip1)

    video_state = _make_video_state()
    result = await _normalise_audio_streams(
        [clip0, clip1], video_dir, video_state=video_state, clip_indices=[0, 1]
    )

    assert result == [clip0, clip1]
    assert not any(video_dir.glob("*_silenced.mp4"))
    assert not any(k.endswith("_silenced") for k in video_state.artifacts)


@pytest.mark.asyncio
async def test_normalise_swaps_in_silenced_paths_when_mixed(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """One clip with audio + one without → silenced variant created, swapped in, and registered."""
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    clip_with_audio = video_dir / "clip_0_trimmed.mp4"
    clip_silent = video_dir / "clip_1_trimmed.mp4"
    await _add_silent_audio_track(clip_fixture_mp4_path, clip_with_audio)
    shutil.copy2(clip_fixture_mp4_path, clip_silent)

    video_state = _make_video_state()
    result = await _normalise_audio_streams(
        [clip_with_audio, clip_silent], video_dir, video_state=video_state, clip_indices=[0, 1]
    )

    # First clip (had audio) unchanged
    assert result[0] == clip_with_audio
    # Second clip (was silent) swapped for _silenced variant
    assert result[1] != clip_silent
    assert "_silenced" in result[1].name
    assert result[1].exists()
    assert ffprobe_has_audio(result[1]) is True
    # Artifact registered for the silenced clip (index 1)
    assert "clip_1_silenced" in video_state.artifacts
    silenced_path = pathlib.Path(video_state.artifacts["clip_1_silenced"])
    assert silenced_path.exists()
    # No artifact for the clip that already had audio (index 0)
    assert "clip_0_silenced" not in video_state.artifacts


# ---------------------------------------------------------------------------
# Audio normalisation — via run_stitch integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalise_skipped_when_all_silent(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """All-silent clips: run_stitch succeeds, no _silenced files created."""
    video_state = _make_video_state()
    _setup_two_trimmed_clips(clip_fixture_mp4_path, tmp_path, video_state)

    out_path = await run_stitch(
        video_state,
        artifacts_root=tmp_path,
        cleanup_raw=False,
        cleanup_trimmed=False,
    )

    assert out_path.exists()
    assert ffprobe_has_audio(out_path) is False
    video_dir = tmp_path / video_state.video_id
    assert not any(video_dir.glob("*_silenced.mp4"))


@pytest.mark.asyncio
async def test_normalise_skipped_when_all_have_audio(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """All-audio clips: run_stitch succeeds, no _silenced files, output has audio."""
    video_state = _make_video_state()
    video_dir = tmp_path / video_state.video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    for i in range(2):
        dst = video_dir / f"clip_{i}_trimmed.mp4"
        await _add_silent_audio_track(clip_fixture_mp4_path, dst)
        video_state.artifacts[f"clip_{i}_trimmed"] = str(dst)

    out_path = await run_stitch(
        video_state,
        artifacts_root=tmp_path,
        cleanup_raw=False,
        cleanup_trimmed=False,
    )

    assert out_path.exists()
    assert ffprobe_has_audio(out_path) is True
    assert not any(video_dir.glob("*_silenced.mp4"))


@pytest.mark.asyncio
async def test_normalise_adds_silent_track_when_mixed(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Mixed audio clips: _silenced file created, stitched output has audio."""
    video_state = _make_video_state()
    video_dir = tmp_path / video_state.video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    # clip_0: has audio; clip_1: silent
    clip0 = video_dir / "clip_0_trimmed.mp4"
    clip1 = video_dir / "clip_1_trimmed.mp4"
    await _add_silent_audio_track(clip_fixture_mp4_path, clip0)
    shutil.copy2(clip_fixture_mp4_path, clip1)
    video_state.artifacts["clip_0_trimmed"] = str(clip0)
    video_state.artifacts["clip_1_trimmed"] = str(clip1)

    out_path = await run_stitch(
        video_state,
        artifacts_root=tmp_path,
        cleanup_raw=False,
        cleanup_trimmed=False,
    )

    assert out_path.exists()
    assert ffprobe_has_audio(out_path) is True
    silenced = list(video_dir.glob("*_silenced.mp4"))
    assert len(silenced) == 1, f"Expected 1 _silenced.mp4, found: {silenced}"

    # The silenced intermediate must be registered in artifacts (clip_1 was the silent one)
    assert "clip_1_silenced" in video_state.artifacts
    silenced_artifact_path = pathlib.Path(video_state.artifacts["clip_1_silenced"])
    assert silenced_artifact_path.exists()

    # Duration should be ~16 s (two 8 s clips)
    duration = ffprobe_duration_seconds(out_path)
    assert abs(duration - 16.0) < 1.0, f"Unexpected stitched duration: {duration}"


# ---------------------------------------------------------------------------
# Bug #4 — _silenced.mp4 cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_silenced_intermediate_cleaned_up_with_trimmed_clips(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """cleanup_trimmed=True (default) must delete _silenced files and their artifact keys."""
    video_state = _make_video_state()
    video_dir = tmp_path / video_state.video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    # clip_0: has audio; clip_1: silent → will produce clip_1_silenced
    clip0 = video_dir / "clip_0_trimmed.mp4"
    clip1 = video_dir / "clip_1_trimmed.mp4"
    await _add_silent_audio_track(clip_fixture_mp4_path, clip0)
    shutil.copy2(clip_fixture_mp4_path, clip1)
    video_state.artifacts["clip_0_trimmed"] = str(clip0)
    video_state.artifacts["clip_1_trimmed"] = str(clip1)

    await run_stitch(
        video_state,
        artifacts_root=tmp_path,
        cleanup_raw=False,
        cleanup_trimmed=True,  # default — should also clean _silenced
    )

    # _silenced artifact key removed
    assert not any(k.endswith("_silenced") for k in video_state.artifacts)
    # _silenced file deleted from disk
    assert not any(video_dir.glob("*_silenced.mp4"))
    # _trimmed keys also gone
    assert not any(k.endswith("_trimmed") for k in video_state.artifacts)


@pytest.mark.asyncio
async def test_silenced_intermediate_kept_when_cleanup_trimmed_false(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """cleanup_trimmed=False must leave _silenced files and their artifact keys intact."""
    video_state = _make_video_state()
    video_dir = tmp_path / video_state.video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    clip0 = video_dir / "clip_0_trimmed.mp4"
    clip1 = video_dir / "clip_1_trimmed.mp4"
    await _add_silent_audio_track(clip_fixture_mp4_path, clip0)
    shutil.copy2(clip_fixture_mp4_path, clip1)
    video_state.artifacts["clip_0_trimmed"] = str(clip0)
    video_state.artifacts["clip_1_trimmed"] = str(clip1)

    await run_stitch(
        video_state,
        artifacts_root=tmp_path,
        cleanup_raw=False,
        cleanup_trimmed=False,
    )

    # _silenced artifact key present
    assert "clip_1_silenced" in video_state.artifacts
    silenced_path = pathlib.Path(video_state.artifacts["clip_1_silenced"])
    assert silenced_path.exists()


def test_cleanup_after_step_removes_silenced_with_trimmed(tmp_path: pathlib.Path) -> None:
    """cleanup_after_step with keep_trimmed_clips=False must also delete _silenced artifacts."""
    video_state = _make_video_state()

    for i in range(2):
        for suffix in ("_raw", "_trimmed", "_silenced"):
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
    assert not any(k.endswith("_silenced") for k in video_state.artifacts)
    # All files deleted from disk
    for i in range(2):
        for suffix in ("_raw", "_trimmed", "_silenced"):
            assert not (tmp_path / f"clip_{i}{suffix}.mp4").exists()


def test_cleanup_after_step_keeps_silenced_when_keep_trimmed_true(tmp_path: pathlib.Path) -> None:
    """keep_trimmed_clips=True must also preserve _silenced artifacts."""
    video_state = _make_video_state()

    silenced = tmp_path / "clip_0_silenced.mp4"
    silenced.write_bytes(b"dummy")
    video_state.artifacts["clip_0_silenced"] = str(silenced)

    cleanup_after_step(
        video_state,
        "stitch",
        {"keep_raw_clips": False, "keep_trimmed_clips": True},
    )

    assert "clip_0_silenced" in video_state.artifacts
    assert silenced.exists()


# ---------------------------------------------------------------------------
# _build_xfade_args — pure-function unit tests (no ffmpeg needed)
# ---------------------------------------------------------------------------


def test_build_xfade_args_two_clips_with_audio(tmp_path: pathlib.Path) -> None:
    """2 clips × 8 s, transition 0.4 s — verify xfade offset and arg structure."""
    clip0 = tmp_path / "clip_0.mp4"
    clip1 = tmp_path / "clip_1.mp4"
    clip0.touch()
    clip1.touch()
    out = tmp_path / "stitched.mp4"

    args = _build_xfade_args(
        clip_paths=[clip0, clip1],
        clip_durations=[8.0, 8.0],
        has_audio=True,
        transition_seconds=0.4,
        out_path=out,
    )

    # Verify -i flags
    assert "-i" in args
    assert str(clip0) in args
    assert str(clip1) in args

    # Verify filter_complex contains xfade with correct offset (8.0 - 0.4*1 = 7.6)
    fc_index = args.index("-filter_complex")
    fc_value = args[fc_index + 1]
    assert "xfade=transition=fade:duration=0.4:offset=7.6" in fc_value

    # Verify acrossfade is included when has_audio=True
    assert "acrossfade=d=0.4" in fc_value

    # Verify output path and encode flags
    assert str(out) in args
    assert "libx264" in args
    assert "aac" in args


def test_build_xfade_args_two_clips_no_audio(tmp_path: pathlib.Path) -> None:
    """2 clips, no audio — acrossfade must be absent; -c:a must be absent."""
    clip0 = tmp_path / "clip_0.mp4"
    clip1 = tmp_path / "clip_1.mp4"
    clip0.touch()
    clip1.touch()
    out = tmp_path / "stitched.mp4"

    args = _build_xfade_args(
        clip_paths=[clip0, clip1],
        clip_durations=[8.0, 8.0],
        has_audio=False,
        transition_seconds=0.4,
        out_path=out,
    )

    fc_index = args.index("-filter_complex")
    fc_value = args[fc_index + 1]

    # xfade present, acrossfade absent
    assert "xfade" in fc_value
    assert "acrossfade" not in fc_value

    # No -c:a flag at all
    assert "-c:a" not in args
    assert "aac" not in args


def test_build_xfade_args_three_clips_offset_formula(tmp_path: pathlib.Path) -> None:
    """3 clips × 8 s, transition 0.3 s — verify both xfade offsets."""
    clips = [tmp_path / f"clip_{i}.mp4" for i in range(3)]
    for c in clips:
        c.touch()
    out = tmp_path / "stitched.mp4"

    args = _build_xfade_args(
        clip_paths=clips,
        clip_durations=[8.0, 8.0, 8.0],
        has_audio=False,
        transition_seconds=0.3,
        out_path=out,
    )

    fc_index = args.index("-filter_complex")
    fc_value = args[fc_index + 1]

    # First xfade: offset = 8.0 - 0.3*1 = 7.7
    assert "xfade=transition=fade:duration=0.3:offset=7.7" in fc_value
    # Second xfade: offset = (8.0 + 8.0) - 0.3*2 = 15.4
    assert "xfade=transition=fade:duration=0.3:offset=15.4" in fc_value


# ---------------------------------------------------------------------------
# Crossfade integration tests (real ffmpeg, real fixture clips)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_stitch_with_crossfade_two_clips(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """2 clips × 8 s with 0.4 s crossfade → expected duration ~15.6 s (±0.5 s)."""
    video_state = _make_video_state()
    _setup_two_trimmed_clips(clip_fixture_mp4_path, tmp_path, video_state)

    out_path = await run_stitch(
        video_state,
        artifacts_root=tmp_path,
        cleanup_raw=False,
        cleanup_trimmed=False,
        transition_seconds=0.4,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0

    duration = ffprobe_duration_seconds(out_path)
    expected = 8.0 + 8.0 - 0.4
    assert abs(duration - expected) < 0.5, (
        f"Expected ~{expected}s, got {duration}s"
    )

    assert video_state.artifacts.get("stitched") == str(out_path)


@pytest.mark.asyncio
async def test_run_stitch_with_crossfade_three_clips(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """3 clips × 8 s with 0.3 s crossfade → expected duration ~23.4 s (±0.5 s)."""
    video_state = _make_video_state()
    video_dir = tmp_path / video_state.video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        dst = video_dir / f"clip_{i}_trimmed.mp4"
        shutil.copy2(clip_fixture_mp4_path, dst)
        video_state.artifacts[f"clip_{i}_trimmed"] = str(dst)

    out_path = await run_stitch(
        video_state,
        artifacts_root=tmp_path,
        cleanup_raw=False,
        cleanup_trimmed=False,
        transition_seconds=0.3,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0

    duration = ffprobe_duration_seconds(out_path)
    expected = 8.0 * 3 - 0.3 * 2
    assert abs(duration - expected) < 0.5, (
        f"Expected ~{expected}s, got {duration}s"
    )


@pytest.mark.asyncio
async def test_run_stitch_with_crossfade_no_audio_skips_acrossfade(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Silent clips + crossfade → output has no audio stream, correct video duration."""
    # clip_fixture_mp4_path is already silent (no audio stream)
    video_state = _make_video_state()
    _setup_two_trimmed_clips(clip_fixture_mp4_path, tmp_path, video_state)

    out_path = await run_stitch(
        video_state,
        artifacts_root=tmp_path,
        cleanup_raw=False,
        cleanup_trimmed=False,
        transition_seconds=0.4,
    )

    assert out_path.exists()
    assert ffprobe_has_audio(out_path) is False

    duration = ffprobe_duration_seconds(out_path)
    expected = 8.0 + 8.0 - 0.4
    assert abs(duration - expected) < 0.5, (
        f"Expected ~{expected}s, got {duration}s"
    )


@pytest.mark.asyncio
async def test_run_stitch_zero_transition_uses_concat_path(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """transition_seconds=0.0 (default) uses the fast concat-demuxer path → ~16 s output."""
    video_state = _make_video_state()
    _setup_two_trimmed_clips(clip_fixture_mp4_path, tmp_path, video_state)

    out_path = await run_stitch(
        video_state,
        artifacts_root=tmp_path,
        cleanup_raw=False,
        cleanup_trimmed=False,
        transition_seconds=0.0,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0

    duration = ffprobe_duration_seconds(out_path)
    assert abs(duration - 16.0) < 0.5, f"Unexpected duration: {duration}"

    # concat demuxer writes a concat_list.txt; xfade path does not
    video_dir = tmp_path / video_state.video_id
    assert (video_dir / "concat_list.txt").exists()
