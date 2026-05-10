"""Unit tests for the first_frame_composite step (mocked Nano Banana client).

All tests are fully offline — no real API calls are made.

Cost billing note (SPEC.md §12): cost is recorded BEFORE each API call
attempt. For the safety retry, both the original attempt and the softened
retry each bill once, giving 2x the per-image cost. For the tenacity-wrapped
transient retry, each attempt bills once — so 3 failed attempts (2 errors +
1 success) would bill 3 times.
"""

from __future__ import annotations

import pathlib
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from ugc_pipeline.models import ProductBrief, RunState, VideoSpec, VideoState
from ugc_pipeline.prompts.first_frame import REFERENCE_PREAMBLE
from ugc_pipeline.steps.first_frame import (
    NanoBananaGenerationError,
    NanoBananaResult,
    NanoBananaSafetyError,
    estimate_first_frame_cost_usd,
    run_first_frame_for_clip,
    run_first_frames,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # minimal valid-ish PNG header


def _make_brief(**overrides: object) -> ProductBrief:
    defaults: dict = {
        "product_id": "a3f9c12e7b04",
        "image_path": "artifacts/a3f9c12e7b04/product.jpg",
        "shape": "cylindrical mug with a C-shaped handle",
        "dominant_colours": ["#F5F0E8", "#3B2A1A"],
        "packaging_style": "kraft paper box with embossed geometric pattern and ribbon closure",
        "inferred_category": "kitchenware / home lifestyle",
        "lifestyle_contexts": [
            "morning routine at a kitchen counter",
            "desk setup during remote work",
        ],
        "visual_notes": "matte ceramic surface with slight speckle texture",
    }
    defaults.update(overrides)
    return ProductBrief(**defaults)  # type: ignore[arg-type]


def _make_spec(clip_count: int = 2, **overrides: object) -> VideoSpec:
    vid = str(uuid.uuid4())
    defaults: dict = {
        "video_id": vid,
        "product_id": "a3f9c12e7b04",
        "spec_index": 0,
        "tone": "warm storyteller",
        "narrative_arc": "A quiet narrative arc.",
        "talent_id": "talent_01",
        "clip_count": clip_count,
        "scene_descriptions": [f"Scene {i} description." for i in range(clip_count)],
        "script_blocks": ["" if i != 1 else f"Testo italiano per clip {i}." for i in range(clip_count)],
        "visual_style_notes": "Warm amber tone grading.",
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return VideoSpec(**defaults)  # type: ignore[arg-type]


def _make_video_state(spec: VideoSpec) -> VideoState:
    return VideoState(
        video_id=spec.video_id,
        product_id=spec.product_id,
        spec_index=spec.spec_index,
        status="in_progress",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _make_run_state() -> RunState:
    return RunState(
        run_id=str(uuid.uuid4()),
        started_at=datetime.now(timezone.utc),
    )


def _make_mock_client(png_bytes: bytes = _FAKE_PNG) -> MagicMock:
    """Return a mock client whose generate_image returns a NanaBananaResult."""
    client = MagicMock()
    client.generate_image = AsyncMock(
        return_value=NanoBananaResult(png_bytes=png_bytes, model="gemini-3-flash-image")
    )
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path(tmp_path: pathlib.Path) -> None:
    """Test 1: Happy path — generates PNG, file exists, artifacts updated, cost tracked."""
    spec = _make_spec(clip_count=2)
    brief = _make_brief()
    video_state = _make_video_state(spec)
    run_state = _make_run_state()
    client = _make_mock_client()

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"

    out_path = await run_first_frame_for_clip(
        spec=spec,
        brief=brief,
        clip_index=0,
        talent_descriptor="woman, late 20s, relaxed aesthetic",
        client=client,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        global_max_usd=50.0,
    )

    # File must exist with the expected name
    assert out_path.exists(), f"Expected PNG at {out_path}"
    assert out_path.stat().st_size > 0

    # Artifacts dict must be updated
    assert "clip_0_firstframe" in video_state.artifacts
    assert video_state.artifacts["clip_0_firstframe"] == str(out_path.resolve())

    # Cost must be tracked
    expected_cost = estimate_first_frame_cost_usd()
    assert video_state.costs_usd.first_frame_usd == pytest.approx(expected_cost)
    assert run_state.cumulative_cost_usd == pytest.approx(expected_cost)

    # Client must have been called exactly once
    client.generate_image.assert_awaited_once()


@pytest.mark.asyncio
async def test_idempotency(tmp_path: pathlib.Path) -> None:
    """Test 2: Second call with existing artifact does NOT call client; state unchanged."""
    spec = _make_spec(clip_count=2)
    brief = _make_brief()
    video_state = _make_video_state(spec)
    run_state = _make_run_state()
    client = _make_mock_client()

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"

    # Create the artifact file manually to simulate prior successful run
    artifact_key = "clip_0_firstframe"
    existing_png = artifacts_root / spec.video_id / "clip_0_firstframe.png"
    existing_png.parent.mkdir(parents=True, exist_ok=True)
    existing_png.write_bytes(_FAKE_PNG)
    video_state.artifacts[artifact_key] = str(existing_png)

    cost_before = video_state.costs_usd.first_frame_usd
    cumulative_before = run_state.cumulative_cost_usd

    out_path = await run_first_frame_for_clip(
        spec=spec,
        brief=brief,
        clip_index=0,
        talent_descriptor="woman, late 20s, relaxed aesthetic",
        client=client,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        global_max_usd=50.0,
    )

    # No API call
    client.generate_image.assert_not_awaited()

    # Cost unchanged
    assert video_state.costs_usd.first_frame_usd == pytest.approx(cost_before)
    assert run_state.cumulative_cost_usd == pytest.approx(cumulative_before)

    # Returns the existing path
    assert out_path == existing_png


@pytest.mark.asyncio
async def test_dry_run(tmp_path: pathlib.Path) -> None:
    """Test 3: Dry-run — no client call, no file written, returns expected path."""
    spec = _make_spec(clip_count=2)
    brief = _make_brief()
    video_state = _make_video_state(spec)
    run_state = _make_run_state()
    client = _make_mock_client()

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"

    out_path = await run_first_frame_for_clip(
        spec=spec,
        brief=brief,
        clip_index=1,
        talent_descriptor="man, early 30s, energetic",
        client=client,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        global_max_usd=50.0,
        dry_run=True,
    )

    # No API call
    client.generate_image.assert_not_awaited()

    # File must NOT exist (dry-run skips creation)
    assert not out_path.exists()

    # No cost tracked
    assert video_state.costs_usd.first_frame_usd == pytest.approx(0.0)
    assert run_state.cumulative_cost_usd == pytest.approx(0.0)

    # Path has expected structure
    assert out_path.name == "clip_1_firstframe.png"


@pytest.mark.asyncio
async def test_safety_retry_success(tmp_path: pathlib.Path) -> None:
    """Test 4: Safety retry — first call raises NanoBananaSafetyError, second (softened) succeeds.

    Assert client.generate_image called twice; second call uses softened prompt
    (contains the safety preamble prefix).
    """
    spec = _make_spec(clip_count=2)
    brief = _make_brief()
    video_state = _make_video_state(spec)
    run_state = _make_run_state()

    client = MagicMock()
    client.generate_image = AsyncMock(
        side_effect=[
            NanoBananaSafetyError("safety block"),
            NanoBananaResult(png_bytes=_FAKE_PNG, model="gemini-3-flash-image"),
        ]
    )

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"

    out_path = await run_first_frame_for_clip(
        spec=spec,
        brief=brief,
        clip_index=0,
        talent_descriptor="woman, late 20s, relaxed aesthetic",
        client=client,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        global_max_usd=50.0,
    )

    # Called twice total
    assert client.generate_image.await_count == 2

    # Second call must use the softened prompt (contains the safety preamble)
    second_call_kwargs = client.generate_image.await_args_list[1].kwargs
    second_prompt = second_call_kwargs["prompt"]
    assert second_prompt.startswith(
        "A tasteful, brand-safe still image with no people in close physical contact:"
    ), f"Expected softened prompt prefix; got: {second_prompt[:120]!r}"

    # File written and artifact recorded
    assert out_path.exists()
    assert video_state.artifacts["clip_0_firstframe"] == str(out_path.resolve())

    # Cost billed once: bill-after-success protocol means the failed first
    # attempt does not bill; only the successful softened-prompt retry does.
    expected_cost = estimate_first_frame_cost_usd()
    assert video_state.costs_usd.first_frame_usd == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_safety_retry_exhausted(tmp_path: pathlib.Path) -> None:
    """Test 5: Safety retry exhausted — both calls raise NanoBananaSafetyError.

    Exception propagates; video_state.last_error is NOT set here (orchestrator
    responsibility).
    """
    spec = _make_spec(clip_count=2)
    brief = _make_brief()
    video_state = _make_video_state(spec)
    run_state = _make_run_state()

    client = MagicMock()
    client.generate_image = AsyncMock(
        side_effect=NanoBananaSafetyError("persistent safety block")
    )

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"

    with pytest.raises(NanoBananaSafetyError):
        await run_first_frame_for_clip(
            spec=spec,
            brief=brief,
            clip_index=0,
            talent_descriptor="woman, late 20s, relaxed aesthetic",
            client=client,
            video_state=video_state,
            run_state=run_state,
            state_root=state_root,
            artifacts_root=artifacts_root,
            global_max_usd=50.0,
        )

    # Orchestrator sets last_error; step itself does not
    assert video_state.last_error is None

    # No artifact persisted
    assert "clip_0_firstframe" not in video_state.artifacts


@pytest.mark.asyncio
async def test_generation_error_transient_retry(tmp_path: pathlib.Path) -> None:
    """Test 6: Generation error transient retry — first 2 calls fail, third succeeds.

    Cost incremented once per attempt (3 attempts = 3x per-image cost).
    This reflects SPEC.md §12: cost is recorded BEFORE each attempt so that
    crashes during any attempt still appear in accounting.
    """
    spec = _make_spec(clip_count=2)
    brief = _make_brief()
    video_state = _make_video_state(spec)
    run_state = _make_run_state()

    client = MagicMock()
    client.generate_image = AsyncMock(
        side_effect=[
            NanoBananaGenerationError("transient error 1"),
            NanoBananaGenerationError("transient error 2"),
            NanoBananaResult(png_bytes=_FAKE_PNG, model="gemini-3-flash-image"),
        ]
    )

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"

    out_path = await run_first_frame_for_clip(
        spec=spec,
        brief=brief,
        clip_index=0,
        talent_descriptor="woman, late 20s, relaxed aesthetic",
        client=client,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        global_max_usd=50.0,
    )

    # Called 3 times total (2 failures + 1 success)
    assert client.generate_image.await_count == 3

    # File written successfully on third attempt
    assert out_path.exists()
    assert video_state.artifacts["clip_0_firstframe"] == str(out_path.resolve())

    # Cost billed only once: bill-after-success protocol — the two failed
    # NanoBananaGenerationError attempts produce no compute and are not billed;
    # only the successful third attempt bills.
    expected_cost = estimate_first_frame_cost_usd()
    assert video_state.costs_usd.first_frame_usd == pytest.approx(expected_cost)
    assert run_state.cumulative_cost_usd == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_run_first_frames_all_clips(tmp_path: pathlib.Path) -> None:
    """run_first_frames orchestrates all clips for a spec in order."""
    clip_count = 3
    spec = _make_spec(clip_count=clip_count)
    brief = _make_brief()
    video_state = _make_video_state(spec)
    run_state = _make_run_state()
    client = _make_mock_client()

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"

    paths = await run_first_frames(
        spec=spec,
        brief=brief,
        talent_descriptor="man, early 30s, energetic",
        client=client,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
    )

    assert len(paths) == clip_count
    assert client.generate_image.await_count == clip_count

    for i, path in enumerate(paths):
        assert path.name == f"clip_{i}_firstframe.png"
        assert path.exists()

    # All artifacts recorded
    for i in range(clip_count):
        assert f"clip_{i}_firstframe" in video_state.artifacts


# ---------------------------------------------------------------------------
# Reference-image tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_first_frame_for_clip_passes_reference_bytes(tmp_path: pathlib.Path) -> None:
    """When reference_image_bytes is supplied, generate_image is called with it."""
    spec = _make_spec(clip_count=2)
    brief = _make_brief()
    video_state = _make_video_state(spec)
    run_state = _make_run_state()
    client = _make_mock_client()

    ref_bytes = b"\x89PNG\r\n\x1a\nFAKE"

    await run_first_frame_for_clip(
        spec=spec,
        brief=brief,
        clip_index=1,
        talent_descriptor="woman, late 20s, relaxed aesthetic",
        client=client,
        video_state=video_state,
        run_state=run_state,
        state_root=tmp_path / "state",
        artifacts_root=tmp_path / "artifacts",
        reference_image_bytes=ref_bytes,
    )

    client.generate_image.assert_awaited_once()
    call_kwargs = client.generate_image.await_args.kwargs
    assert call_kwargs["reference_image_bytes"] == ref_bytes


@pytest.mark.asyncio
async def test_run_first_frame_for_clip_no_reference_keeps_text_only(tmp_path: pathlib.Path) -> None:
    """When reference_image_bytes is not supplied, generate_image is called with None."""
    spec = _make_spec(clip_count=2)
    brief = _make_brief()
    video_state = _make_video_state(spec)
    run_state = _make_run_state()
    client = _make_mock_client()

    await run_first_frame_for_clip(
        spec=spec,
        brief=brief,
        clip_index=0,
        talent_descriptor="woman, late 20s, relaxed aesthetic",
        client=client,
        video_state=video_state,
        run_state=run_state,
        state_root=tmp_path / "state",
        artifacts_root=tmp_path / "artifacts",
    )

    client.generate_image.assert_awaited_once()
    call_kwargs = client.generate_image.await_args.kwargs
    assert call_kwargs["reference_image_bytes"] is None


@pytest.mark.asyncio
async def test_run_first_frames_chains_clip0_bytes_to_subsequent_clips(tmp_path: pathlib.Path) -> None:
    """run_first_frames feeds clip-0 PNG bytes as reference_image_bytes to clips 1+."""
    clip_count = 3
    spec = _make_spec(clip_count=clip_count)
    brief = _make_brief()
    video_state = _make_video_state(spec)
    run_state = _make_run_state()

    # Each call returns the same _FAKE_PNG; clip-0 bytes will be _FAKE_PNG
    client = _make_mock_client(png_bytes=_FAKE_PNG)

    await run_first_frames(
        spec=spec,
        brief=brief,
        talent_descriptor="man, early 30s, energetic",
        client=client,
        video_state=video_state,
        run_state=run_state,
        state_root=tmp_path / "state",
        artifacts_root=tmp_path / "artifacts",
    )

    assert client.generate_image.await_count == clip_count

    calls = client.generate_image.await_args_list

    # Clip 0: no reference
    assert calls[0].kwargs["reference_image_bytes"] is None

    # Clips 1 and 2: reference == _FAKE_PNG (the bytes written by clip 0)
    assert calls[1].kwargs["reference_image_bytes"] == _FAKE_PNG
    assert calls[2].kwargs["reference_image_bytes"] == _FAKE_PNG


@pytest.mark.asyncio
async def test_run_first_frames_dry_run_passes_no_reference(tmp_path: pathlib.Path) -> None:
    """In dry_run, no reference_image_bytes is ever read or passed; no FileNotFoundError."""
    clip_count = 3
    spec = _make_spec(clip_count=clip_count)
    brief = _make_brief()
    video_state = _make_video_state(spec)
    run_state = _make_run_state()
    client = _make_mock_client()

    # Must not raise FileNotFoundError (path.read_bytes() skipped in dry_run)
    paths = await run_first_frames(
        spec=spec,
        brief=brief,
        talent_descriptor="man, early 30s, energetic",
        client=client,
        video_state=video_state,
        run_state=run_state,
        state_root=tmp_path / "state",
        artifacts_root=tmp_path / "artifacts",
        dry_run=True,
    )

    # dry_run skips all API calls
    client.generate_image.assert_not_awaited()

    # Returns expected paths (non-existent in dry_run)
    assert len(paths) == clip_count
    for path in paths:
        assert not path.exists()


@pytest.mark.asyncio
async def test_safety_retry_keeps_reference_image(tmp_path: pathlib.Path) -> None:
    """Safety retry keeps the reference image; only the text prompt is softened.

    The softened retry prompt must include the REFERENCE_PREAMBLE (since the
    reference image was provided), confirming render_with_reference_softened was used.
    """
    spec = _make_spec(clip_count=2)
    brief = _make_brief()
    video_state = _make_video_state(spec)
    run_state = _make_run_state()

    ref_bytes = b"FAKE_PNG_REF"

    client = MagicMock()
    client.generate_image = AsyncMock(
        side_effect=[
            NanoBananaSafetyError("safety block"),
            NanoBananaResult(png_bytes=_FAKE_PNG, model="gemini-3-flash-image"),
        ]
    )

    await run_first_frame_for_clip(
        spec=spec,
        brief=brief,
        clip_index=1,
        talent_descriptor="woman, late 20s, relaxed aesthetic",
        client=client,
        video_state=video_state,
        run_state=run_state,
        state_root=tmp_path / "state",
        artifacts_root=tmp_path / "artifacts",
        reference_image_bytes=ref_bytes,
    )

    assert client.generate_image.await_count == 2

    retry_kwargs = client.generate_image.await_args_list[1].kwargs

    # Reference image preserved on retry
    assert retry_kwargs["reference_image_bytes"] == ref_bytes

    # Text prompt is the softened+reference variant — contains REFERENCE_PREAMBLE
    retry_prompt = retry_kwargs["prompt"]
    assert REFERENCE_PREAMBLE in retry_prompt, (
        f"Expected REFERENCE_PREAMBLE in softened retry prompt; got: {retry_prompt[:200]!r}"
    )
