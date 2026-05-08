"""Unit tests for src/ugc_pipeline/steps/caption.py.

``transcribe_with_whisper`` is MOCKED via monkeypatch throughout this module
so no model download or network access occurs.  ffmpeg (via imageio_ffmpeg)
is invoked for real to burn the subtitle file into the fixture clip.
"""

from __future__ import annotations

import pathlib
import shutil
import uuid
from datetime import datetime, timezone

import pytest

from ugc_pipeline.models import CostBreakdown, VideoSpec, VideoState
from ugc_pipeline.steps.caption import run_caption


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


def _make_video_spec(video_id: str, clip_count: int = 1) -> VideoSpec:
    return VideoSpec(
        video_id=video_id,
        product_id="test_product",
        spec_index=0,
        tone="neutral",
        narrative_arc="Un momento di calma.",
        talent_id="talent_01",
        clip_count=clip_count,
        scene_descriptions=["Scene description."] * clip_count,
        script_blocks=["Testo italiano."] * clip_count,
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def canned_segments() -> list[dict]:
    """Pre-canned transcript segments that stand in for real whisper output."""
    return [
        {
            "start": 0.0,
            "end": 2.5,
            "text": "Questo è un test.",
            "words": [
                {"start": 0.0, "end": 0.5, "word": "Questo"},
                {"start": 0.6, "end": 0.9, "word": "è"},
                {"start": 1.0, "end": 1.2, "word": "un"},
                {"start": 1.3, "end": 1.8, "word": "test."},
            ],
        },
        {
            "start": 2.5,
            "end": 5.0,
            "text": "È una bella giornata.",
            "words": [
                {"start": 2.5, "end": 2.9, "word": "È"},
                {"start": 3.0, "end": 3.2, "word": "una"},
                {"start": 3.3, "end": 3.7, "word": "bella"},
                {"start": 3.8, "end": 4.5, "word": "giornata."},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Happy path: final.mp4 produced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_caption_produces_final_mp4(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
    canned_segments: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_caption should produce final.mp4 and update artifacts."""
    import ugc_pipeline.steps.caption as caption_mod

    monkeypatch.setattr(caption_mod, "transcribe_with_whisper", lambda *a, **kw: canned_segments)

    video_state = _make_video_state()
    spec = _make_video_spec(video_state.video_id)

    video_dir = tmp_path / video_state.video_id
    video_dir.mkdir(parents=True)
    stitched = video_dir / "stitched.mp4"
    shutil.copy2(clip_fixture_mp4_path, stitched)
    video_state.artifacts["stitched"] = str(stitched)

    out_path = await run_caption(
        video_state, spec, artifacts_root=tmp_path, max_chars_per_line=42
    )

    assert out_path.exists(), "final.mp4 should be created on disk"
    assert out_path.stat().st_size > 0, "final.mp4 should not be empty"
    assert video_state.artifacts.get("final") == str(out_path)
    assert video_state.artifacts.get("captions_ass") is not None

    # .ass file should exist
    ass_path = pathlib.Path(video_state.artifacts["captions_ass"])
    assert ass_path.exists()
    content = ass_path.read_text(encoding="utf-8")
    assert "[Events]" in content


# ---------------------------------------------------------------------------
# Burn-in adds bytes: final > stitched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_caption_final_larger_than_stitched(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
    canned_segments: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Burning subtitles into the video re-encodes the video stream.

    The final file is NOT guaranteed to be larger than the stitched source
    (subtitle burn-in re-encodes; codec efficiency varies). We instead check
    that both files exist and have non-zero size.
    """
    import ugc_pipeline.steps.caption as caption_mod

    monkeypatch.setattr(caption_mod, "transcribe_with_whisper", lambda *a, **kw: canned_segments)

    video_state = _make_video_state()
    spec = _make_video_spec(video_state.video_id)

    video_dir = tmp_path / video_state.video_id
    video_dir.mkdir(parents=True)
    stitched = video_dir / "stitched.mp4"
    shutil.copy2(clip_fixture_mp4_path, stitched)
    video_state.artifacts["stitched"] = str(stitched)

    out_path = await run_caption(video_state, spec, artifacts_root=tmp_path)

    assert out_path.exists() and out_path.stat().st_size > 0
    assert stitched.exists() and stitched.stat().st_size > 0


# ---------------------------------------------------------------------------
# Empty transcript: step succeeds with header-only .ass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_caption_empty_transcript_continues(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty transcript should log a warning but produce final.mp4 without error."""
    import ugc_pipeline.steps.caption as caption_mod

    monkeypatch.setattr(caption_mod, "transcribe_with_whisper", lambda *a, **kw: [])

    video_state = _make_video_state()
    spec = _make_video_spec(video_state.video_id)

    video_dir = tmp_path / video_state.video_id
    video_dir.mkdir(parents=True)
    stitched = video_dir / "stitched.mp4"
    shutil.copy2(clip_fixture_mp4_path, stitched)
    video_state.artifacts["stitched"] = str(stitched)

    out_path = await run_caption(video_state, spec, artifacts_root=tmp_path)

    assert out_path.exists()
    ass_path = pathlib.Path(video_state.artifacts["captions_ass"])
    content = ass_path.read_text(encoding="utf-8")
    # Header-only .ass: no Dialogue lines
    assert "[Events]" in content
    assert "Dialogue:" not in content


# ---------------------------------------------------------------------------
# Idempotency: second call skips re-processing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_caption_idempotent(
    clip_fixture_mp4_path: pathlib.Path,
    tmp_path: pathlib.Path,
    canned_segments: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second call with existing final artifact should skip ffmpeg."""
    import ugc_pipeline.steps.caption as caption_mod

    call_count = {"n": 0}

    def _mock_transcribe(*a, **kw):  # type: ignore[return]
        call_count["n"] += 1
        return canned_segments

    monkeypatch.setattr(caption_mod, "transcribe_with_whisper", _mock_transcribe)

    video_state = _make_video_state()
    spec = _make_video_spec(video_state.video_id)

    video_dir = tmp_path / video_state.video_id
    video_dir.mkdir(parents=True)
    stitched = video_dir / "stitched.mp4"
    shutil.copy2(clip_fixture_mp4_path, stitched)
    video_state.artifacts["stitched"] = str(stitched)

    # First call: produces final.mp4
    out1 = await run_caption(video_state, spec, artifacts_root=tmp_path)
    first_mtime = out1.stat().st_mtime
    assert call_count["n"] == 1

    # Second call: should be a no-op
    out2 = await run_caption(video_state, spec, artifacts_root=tmp_path)
    assert out2 == out1
    assert out2.stat().st_mtime == first_mtime
    assert call_count["n"] == 1  # transcribe not called again
