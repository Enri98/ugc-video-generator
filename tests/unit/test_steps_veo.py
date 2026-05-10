"""Unit tests for the veo_generation step (SPEC.md §5 Steps 5 + 6).

All Veo and Gemini Flash calls are mocked — no paid API calls are made.
"""

from __future__ import annotations

import asyncio
import pathlib
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from ugc_pipeline.models import (
    CostBreakdown,
    ProductBrief,
    RunState,
    VideoSpec,
    VideoState,
)
from ugc_pipeline.prompts import veo as veo_prompt
from ugc_pipeline.steps.veo import (
    VeoGenerationError,
    VeoOperation,
    VeoResult,
    VeoSafetyBlockError,
    VeoTimeoutError,
    _DefaultVeoClient,
    drain_inflight_veo,
    estimate_safety_retry_rewrite_cost_usd,
    estimate_veo_clip_cost_usd,
    poll_veo_operation,
    run_safety_retry_for_clip,
    run_veo_for_clip,
    submit_veo_operation,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_spec(video_id: str, clip_count: int = 2) -> VideoSpec:
    return VideoSpec(
        video_id=video_id,
        product_id="prod000000001",
        spec_index=0,
        tone="warm storyteller",
        narrative_arc="A quiet morning transforms into a ritual of comfort.",
        talent_id="talent_01",
        clip_count=clip_count,
        scene_descriptions=[f"Scene {i}: talent holds the product." for i in range(clip_count)],
        script_blocks=["" if i != 1 else f"Blocco {i}: testo italiano." for i in range(clip_count)],
        created_at=datetime.now(timezone.utc),
    )


def _make_brief() -> ProductBrief:
    return ProductBrief(
        product_id="prod000000001",
        image_path="artifacts/prod000000001/product.jpg",
        shape="cylindrical mug",
        dominant_colours=["#F5F0E8"],
        packaging_style="kraft paper box",
        inferred_category="kitchenware",
        lifestyle_contexts=["morning routine"],
        created_at=datetime.now(timezone.utc),
    )


def _make_run_state() -> RunState:
    return RunState(
        run_id=str(uuid.uuid4()),
        started_at=datetime.now(timezone.utc),
    )


def _make_video_state(video_id: str) -> VideoState:
    return VideoState(
        video_id=video_id,
        product_id="prod000000001",
        spec_index=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _make_mock_veo_client(
    operation_id: str = "op-001",
    poll_responses: list[dict] | None = None,
) -> MagicMock:
    """Return a mock VeoClientProtocol with configurable poll sequence."""
    client = MagicMock()
    client.submit = AsyncMock(return_value=operation_id)
    if poll_responses is None:
        poll_responses = [{"done": True, "mp4_bytes": b"FAKE_MP4", "error": None, "safety_block": False}]
    poll_side_effects = list(poll_responses)
    client.poll = AsyncMock(side_effect=poll_side_effects)
    return client


def _make_mock_flash_client(rewrite_return: str = "Rewritten prompt.") -> MagicMock:
    client = MagicMock()
    client.rewrite = AsyncMock(return_value=rewrite_return)
    return client


# ---------------------------------------------------------------------------
# Test 1 — submit_veo_operation: returns operation_id, retries on transient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_veo_operation_returns_op_id():
    client = _make_mock_veo_client(operation_id="op-42")
    result = await submit_veo_operation(client, image_bytes=b"PNG", prompt="test prompt")
    assert result == "op-42"
    client.submit.assert_awaited_once()


@pytest.mark.asyncio
async def test_submit_veo_operation_retries_on_transient():
    client = MagicMock()
    # Fails twice, then succeeds
    client.submit = AsyncMock(side_effect=[RuntimeError("transient"), RuntimeError("transient"), "op-99"])
    result = await submit_veo_operation(client, image_bytes=b"PNG", prompt="test")
    assert result == "op-99"
    assert client.submit.await_count == 3


# ---------------------------------------------------------------------------
# Test 2 — poll_veo_operation: pending then done returns VeoResult
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_veo_operation_pending_then_done():
    client = _make_mock_veo_client(
        poll_responses=[
            {"done": False, "mp4_bytes": None, "error": None, "safety_block": False},
            {"done": True, "mp4_bytes": b"MP4_BYTES", "error": None, "safety_block": False},
        ]
    )
    result = await poll_veo_operation(
        client,
        "op-001",
        poll_interval_seconds=0.001,
        poll_timeout_seconds=10.0,
    )
    assert isinstance(result, VeoResult)
    assert result.mp4_bytes == b"MP4_BYTES"
    assert result.operation_id == "op-001"


# ---------------------------------------------------------------------------
# Test 3 — poll_veo_operation: safety_block raises VeoSafetyBlockError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_veo_operation_safety_block():
    client = _make_mock_veo_client(
        poll_responses=[
            {"done": True, "mp4_bytes": None, "error": None, "safety_block": True},
        ]
    )
    with pytest.raises(VeoSafetyBlockError):
        await poll_veo_operation(
            client,
            "op-blocked",
            poll_interval_seconds=0.001,
            poll_timeout_seconds=10.0,
        )


# ---------------------------------------------------------------------------
# Test 4 — poll_veo_operation: always pending → VeoTimeoutError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_veo_operation_timeout():
    client = MagicMock()
    # Always pending
    client.poll = AsyncMock(
        return_value={"done": False, "mp4_bytes": None, "error": None, "safety_block": False}
    )
    with pytest.raises(VeoTimeoutError):
        await poll_veo_operation(
            client,
            "op-timeout",
            poll_interval_seconds=0.01,
            poll_timeout_seconds=0.05,
        )


# ---------------------------------------------------------------------------
# Test 5 — run_veo_for_clip happy path: writes mp4 to disk, artifact set, cost incremented
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_veo_for_clip_happy_path(tmp_path):
    video_id = str(uuid.uuid4())
    spec = _make_spec(video_id)
    brief = _make_brief()
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"

    # Pre-create a first-frame artifact
    firstframe_dir = artifacts_root / video_id
    firstframe_dir.mkdir(parents=True)
    firstframe_path = firstframe_dir / "clip_0_firstframe.png"
    firstframe_path.write_bytes(b"PNG_BYTES")
    video_state.artifacts["clip_0_firstframe"] = str(firstframe_path)

    # State dir must exist for write_state_atomic
    (state_root / "videos").mkdir(parents=True)
    (state_root / "runs").mkdir(parents=True)

    client = _make_mock_veo_client(
        poll_responses=[{"done": True, "mp4_bytes": b"MP4_DATA", "error": None, "safety_block": False}]
    )

    out_path = await run_veo_for_clip(
        spec,
        brief,
        0,
        client=client,
        flash_client=None,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        poll_interval_seconds=0.001,
        poll_timeout_seconds=10.0,
    )

    assert out_path.exists()
    assert out_path.read_bytes() == b"MP4_DATA"
    assert video_state.artifacts["clip_0_raw"] == str(out_path.resolve())
    assert run_state.cumulative_cost_usd == pytest.approx(estimate_veo_clip_cost_usd())
    assert video_state.costs_usd.veo_usd == pytest.approx(estimate_veo_clip_cost_usd())


# ---------------------------------------------------------------------------
# Test 6 — Idempotency: mp4 exists → no submit call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_veo_for_clip_idempotent(tmp_path):
    video_id = str(uuid.uuid4())
    spec = _make_spec(video_id)
    brief = _make_brief()
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"

    # Pre-create the raw artifact so idempotency triggers
    raw_dir = artifacts_root / video_id
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "clip_0_raw.mp4"
    raw_path.write_bytes(b"ALREADY_EXISTS")
    video_state.artifacts["clip_0_raw"] = str(raw_path)

    client = _make_mock_veo_client()

    out_path = await run_veo_for_clip(
        spec,
        brief,
        0,
        client=client,
        flash_client=None,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        poll_interval_seconds=0.001,
        poll_timeout_seconds=10.0,
    )

    client.submit.assert_not_awaited()
    assert out_path == raw_path


# ---------------------------------------------------------------------------
# Test 7 — Resume from operation_id: state has op:clip_0 but no mp4 → poll only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_veo_for_clip_resume_from_op_id(tmp_path):
    video_id = str(uuid.uuid4())
    spec = _make_spec(video_id)
    brief = _make_brief()
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    (state_root / "videos").mkdir(parents=True)
    (state_root / "runs").mkdir(parents=True)

    # Pre-create first-frame
    firstframe_dir = artifacts_root / video_id
    firstframe_dir.mkdir(parents=True)
    firstframe_path = firstframe_dir / "clip_0_firstframe.png"
    firstframe_path.write_bytes(b"PNG")
    video_state.artifacts["clip_0_firstframe"] = str(firstframe_path)

    # Simulate a stored operation_id (crash during poll)
    video_state.artifacts["op:clip_0"] = "op-existing-001"

    client = _make_mock_veo_client(
        operation_id="op-new-should-not-be-used",
        poll_responses=[{"done": True, "mp4_bytes": b"RESUME_MP4", "error": None, "safety_block": False}],
    )

    out_path = await run_veo_for_clip(
        spec,
        brief,
        0,
        client=client,
        flash_client=None,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        poll_interval_seconds=0.001,
        poll_timeout_seconds=10.0,
    )

    # submit should NOT have been called (we resumed from stored op_id)
    client.submit.assert_not_awaited()
    # poll should have been called with the stored op_id
    client.poll.assert_awaited_once_with("op-existing-001")
    assert out_path.read_bytes() == b"RESUME_MP4"
    # No new cost should have been billed (we resumed)
    assert run_state.cumulative_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Test 8 — Safety retry success: first poll → safety_block; rewrite; second submit+poll → success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_veo_for_clip_safety_retry_success(tmp_path):
    video_id = str(uuid.uuid4())
    spec = _make_spec(video_id)
    brief = _make_brief()
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    (state_root / "videos").mkdir(parents=True)
    (state_root / "runs").mkdir(parents=True)

    # Pre-create first-frame
    firstframe_dir = artifacts_root / video_id
    firstframe_dir.mkdir(parents=True)
    firstframe_path = firstframe_dir / "clip_0_firstframe.png"
    firstframe_path.write_bytes(b"PNG")
    video_state.artifacts["clip_0_firstframe"] = str(firstframe_path)

    submit_counter = {"count": 0}

    async def _submit(**kwargs):
        submit_counter["count"] += 1
        return f"op-{submit_counter['count']:03d}"

    async def _poll(op_id):
        if op_id == "op-001":
            return {"done": True, "mp4_bytes": None, "error": None, "safety_block": True}
        # Second attempt succeeds
        return {"done": True, "mp4_bytes": b"RETRY_MP4", "error": None, "safety_block": False}

    client = MagicMock()
    client.submit = AsyncMock(side_effect=_submit)
    client.poll = AsyncMock(side_effect=_poll)

    flash_client = _make_mock_flash_client(
        "SCENE: Softened safe scene description for the clip.\n\nAUDIO: No spoken dialogue."
    )

    out_path = await run_veo_for_clip(
        spec,
        brief,
        0,
        client=client,
        flash_client=flash_client,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        poll_interval_seconds=0.001,
        poll_timeout_seconds=10.0,
    )

    # Two Veo submits (initial + retry)
    assert client.submit.await_count == 2
    flash_client.rewrite.assert_awaited_once()

    # Two veo_usd increments + one safety_retry_usd
    assert video_state.costs_usd.veo_usd == pytest.approx(estimate_veo_clip_cost_usd() * 2)
    assert video_state.costs_usd.safety_retry_usd == pytest.approx(estimate_safety_retry_rewrite_cost_usd())

    assert out_path.read_bytes() == b"RETRY_MP4"


# ---------------------------------------------------------------------------
# Test 9 — Safety retry failure: both polls → safety_block → raises VeoSafetyBlockError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_veo_for_clip_safety_retry_terminal(tmp_path):
    video_id = str(uuid.uuid4())
    spec = _make_spec(video_id)
    brief = _make_brief()
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    (state_root / "videos").mkdir(parents=True)
    (state_root / "runs").mkdir(parents=True)

    firstframe_dir = artifacts_root / video_id
    firstframe_dir.mkdir(parents=True)
    firstframe_path = firstframe_dir / "clip_0_firstframe.png"
    firstframe_path.write_bytes(b"PNG")
    video_state.artifacts["clip_0_firstframe"] = str(firstframe_path)

    safety_response = {"done": True, "mp4_bytes": None, "error": None, "safety_block": True}

    client = MagicMock()
    client.submit = AsyncMock(side_effect=["op-001", "op-002"])
    client.poll = AsyncMock(return_value=safety_response)

    flash_client = _make_mock_flash_client(
        "SCENE: Still risky rewrite with enough length to pass threshold.\n\nAUDIO: No spoken dialogue."
    )

    with pytest.raises(VeoSafetyBlockError):
        await run_veo_for_clip(
            spec,
            brief,
            0,
            client=client,
            flash_client=flash_client,
            video_state=video_state,
            run_state=run_state,
            state_root=state_root,
            artifacts_root=artifacts_root,
            poll_interval_seconds=0.001,
            poll_timeout_seconds=10.0,
        )


# ---------------------------------------------------------------------------
# Test 10 — Timeout: clip times out → raises VeoTimeoutError, no retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_veo_for_clip_timeout(tmp_path):
    video_id = str(uuid.uuid4())
    spec = _make_spec(video_id)
    brief = _make_brief()
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    (state_root / "videos").mkdir(parents=True)
    (state_root / "runs").mkdir(parents=True)

    firstframe_dir = artifacts_root / video_id
    firstframe_dir.mkdir(parents=True)
    firstframe_path = firstframe_dir / "clip_0_firstframe.png"
    firstframe_path.write_bytes(b"PNG")
    video_state.artifacts["clip_0_firstframe"] = str(firstframe_path)

    client = MagicMock()
    client.submit = AsyncMock(return_value="op-timeout-001")
    client.poll = AsyncMock(
        return_value={"done": False, "mp4_bytes": None, "error": None, "safety_block": False}
    )

    with pytest.raises(VeoTimeoutError):
        await run_veo_for_clip(
            spec,
            brief,
            0,
            client=client,
            flash_client=None,
            video_state=video_state,
            run_state=run_state,
            state_root=state_root,
            artifacts_root=artifacts_root,
            poll_interval_seconds=0.01,
            poll_timeout_seconds=0.05,
        )

    # No second submit attempted
    assert client.submit.await_count == 1


# ---------------------------------------------------------------------------
# Test 11 — Dry-run: no client calls, no file written
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_veo_for_clip_dry_run(tmp_path):
    video_id = str(uuid.uuid4())
    spec = _make_spec(video_id)
    brief = _make_brief()
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"

    client = _make_mock_veo_client()

    out_path = await run_veo_for_clip(
        spec,
        brief,
        0,
        client=client,
        flash_client=None,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        dry_run=True,
    )

    client.submit.assert_not_awaited()
    client.poll.assert_not_awaited()
    assert not out_path.exists()
    assert run_state.cumulative_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Test 12 — Kill-switch drain: drain_inflight_veo polls ops to completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_inflight_veo_polls_to_completion(tmp_path):
    video_id = str(uuid.uuid4())
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    (state_root / "videos").mkdir(parents=True)
    (state_root / "runs").mkdir(parents=True)

    poll_calls = {"count": 0}

    async def _poll(op_id):
        poll_calls["count"] += 1
        if poll_calls["count"] < 2:
            return {"done": False, "mp4_bytes": None, "error": None, "safety_block": False}
        return {"done": True, "mp4_bytes": b"DRAINED_MP4", "error": None, "safety_block": False}

    client = MagicMock()
    client.poll = AsyncMock(side_effect=_poll)

    operations = [
        VeoOperation(operation_id="op-drain-001", clip_index=0, video_id=video_id)
    ]

    await drain_inflight_veo(
        operations=operations,
        client=client,
        run_state=run_state,
        video_states_by_video_id={video_id: video_state},
        state_root=state_root,
        artifacts_root=artifacts_root,
        poll_interval_seconds=0.001,
        poll_timeout_seconds=10.0,
    )

    # MP4 should be on disk
    out_path = artifacts_root / video_id / "clip_0_raw.mp4"
    assert out_path.exists()
    assert out_path.read_bytes() == b"DRAINED_MP4"

    # Artifact recorded in state
    assert video_state.artifacts.get("clip_0_raw") == str(out_path)

    # No new submissions — client has no submit AsyncMock so nothing was awaited
    # MagicMock auto-creates submit as a regular MagicMock; it was not called
    client.submit.assert_not_called()


# ---------------------------------------------------------------------------
# Test 13 — Speaking clip uses AUDIO prompt with Italian script_block
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_veo_speaking_clip_uses_audio_prompt(tmp_path):
    """Clip at speaking_clip_index (default=1) should have AUDIO + Italian script."""
    video_id = str(uuid.uuid4())
    # clip_count=2, speaking_clip_index=1 (default), script_blocks[1] is the spoken line
    spec = _make_spec(video_id, clip_count=2)
    brief = _make_brief()
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    (state_root / "videos").mkdir(parents=True)
    (state_root / "runs").mkdir(parents=True)

    # Pre-create first-frame for clip 1
    firstframe_dir = artifacts_root / video_id
    firstframe_dir.mkdir(parents=True)
    firstframe_path = firstframe_dir / "clip_1_firstframe.png"
    firstframe_path.write_bytes(b"PNG_BYTES")
    video_state.artifacts["clip_1_firstframe"] = str(firstframe_path)

    submitted_prompts: list[str] = []

    async def _capture_submit(**kwargs):
        submitted_prompts.append(kwargs["prompt"])
        return "op-speaking-001"

    client = MagicMock()
    client.submit = AsyncMock(side_effect=_capture_submit)
    client.poll = AsyncMock(
        return_value={"done": True, "mp4_bytes": b"MP4_SPEAKING", "error": None, "safety_block": False}
    )

    await run_veo_for_clip(
        spec,
        brief,
        1,  # speaking clip index
        client=client,
        flash_client=None,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        poll_interval_seconds=0.001,
        poll_timeout_seconds=10.0,
    )

    assert len(submitted_prompts) == 1
    prompt = submitted_prompts[0]
    assert "AUDIO:" in prompt
    # The Italian script_block from _make_spec for clip 1 is "Blocco 1: testo italiano."
    assert "testo italiano" in prompt


# ---------------------------------------------------------------------------
# Test 14 — Silent clip uses ambient-only AUDIO (no script_block content)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_veo_silent_clip_uses_silent_prompt(tmp_path):
    """Clip 0 (non-speaking) should have AUDIO with 'no spoken dialogue', no script."""
    video_id = str(uuid.uuid4())
    spec = _make_spec(video_id, clip_count=2)
    brief = _make_brief()
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    (state_root / "videos").mkdir(parents=True)
    (state_root / "runs").mkdir(parents=True)

    firstframe_dir = artifacts_root / video_id
    firstframe_dir.mkdir(parents=True)
    firstframe_path = firstframe_dir / "clip_0_firstframe.png"
    firstframe_path.write_bytes(b"PNG_BYTES")
    video_state.artifacts["clip_0_firstframe"] = str(firstframe_path)

    submitted_prompts: list[str] = []

    async def _capture_submit(**kwargs):
        submitted_prompts.append(kwargs["prompt"])
        return "op-silent-001"

    client = MagicMock()
    client.submit = AsyncMock(side_effect=_capture_submit)
    client.poll = AsyncMock(
        return_value={"done": True, "mp4_bytes": b"MP4_SILENT", "error": None, "safety_block": False}
    )

    await run_veo_for_clip(
        spec,
        brief,
        0,  # silent clip
        client=client,
        flash_client=None,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        poll_interval_seconds=0.001,
        poll_timeout_seconds=10.0,
    )

    assert len(submitted_prompts) == 1
    prompt = submitted_prompts[0]
    assert "AUDIO:" in prompt
    assert "No spoken dialogue" in prompt
    # The Italian script for clip 1 must NOT appear in clip 0's prompt
    assert "testo italiano" not in prompt


# ---------------------------------------------------------------------------
# Test 15 — Safety retry: second submit is exactly what Flash returned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_safety_retry_reconstructs_full_prompt_with_audio(tmp_path):
    """After safety retry, the second submit should be exactly what Flash returned.

    Flash now rewrites the full composite (SCENE + AUDIO), so the second submit
    prompt equals Flash's output verbatim — no reconstruction via veo_prompt.render.
    The Flash return value here includes SCENE: and AUDIO: markers to verify the
    full composite was passed to Flash and returned as-is.
    """
    video_id = str(uuid.uuid4())
    # Use clip 1 (speaking clip) so the AUDIO section includes the script_block
    spec = _make_spec(video_id, clip_count=2)
    brief = _make_brief()
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    (state_root / "videos").mkdir(parents=True)
    (state_root / "runs").mkdir(parents=True)

    firstframe_dir = artifacts_root / video_id
    firstframe_dir.mkdir(parents=True)
    firstframe_path = firstframe_dir / "clip_1_firstframe.png"
    firstframe_path.write_bytes(b"PNG_BYTES")
    video_state.artifacts["clip_1_firstframe"] = str(firstframe_path)

    submit_counter = {"count": 0}
    submitted_prompts: list[str] = []

    async def _submit(**kwargs):
        submit_counter["count"] += 1
        submitted_prompts.append(kwargs["prompt"])
        return f"op-{submit_counter['count']:03d}"

    async def _poll(op_id):
        if op_id == "op-001":
            return {"done": True, "mp4_bytes": None, "error": None, "safety_block": True}
        return {"done": True, "mp4_bytes": b"RETRY_MP4", "error": None, "safety_block": False}

    client = MagicMock()
    client.submit = AsyncMock(side_effect=_submit)
    client.poll = AsyncMock(side_effect=_poll)

    # Flash returns a full rewritten composite (as the new flow requires)
    flash_return = "SCENE: Safe rewritten scene.\n\nAUDIO: Testo italiano sicuro per la clip."
    flash_client = _make_mock_flash_client(flash_return)

    await run_veo_for_clip(
        spec,
        brief,
        1,  # speaking clip
        client=client,
        flash_client=flash_client,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        poll_interval_seconds=0.001,
        poll_timeout_seconds=10.0,
    )

    assert len(submitted_prompts) == 2
    retry_prompt = submitted_prompts[1]
    # The second submit must be exactly what Flash returned — no reconstruction
    assert retry_prompt == flash_return


# ---------------------------------------------------------------------------
# Test 18 — prompt_versions["veo"] updated after safety retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_versions_veo_updated_after_safety_retry(tmp_path):
    """After safety retry, prompt_versions['veo'].content_sha256 must match the
    SHA of the actually-submitted rewritten prompt (Flash's output), not the
    SHA of the original blocked prompt."""
    import hashlib

    video_id = str(uuid.uuid4())
    spec = _make_spec(video_id, clip_count=2)
    brief = _make_brief()
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    (state_root / "videos").mkdir(parents=True)
    (state_root / "runs").mkdir(parents=True)

    firstframe_dir = artifacts_root / video_id
    firstframe_dir.mkdir(parents=True)
    firstframe_path = firstframe_dir / "clip_0_firstframe.png"
    firstframe_path.write_bytes(b"PNG_BYTES")
    video_state.artifacts["clip_0_firstframe"] = str(firstframe_path)

    submit_counter = {"count": 0}

    async def _submit(**kwargs):
        submit_counter["count"] += 1
        return f"op-{submit_counter['count']:03d}"

    async def _poll(op_id):
        if op_id == "op-001":
            return {"done": True, "mp4_bytes": None, "error": None, "safety_block": True}
        return {"done": True, "mp4_bytes": b"RETRY_MP4", "error": None, "safety_block": False}

    client = MagicMock()
    client.submit = AsyncMock(side_effect=_submit)
    client.poll = AsyncMock(side_effect=_poll)

    flash_return = "SCENE: Softened safe scene description.\n\nAUDIO: No spoken dialogue."
    flash_client = _make_mock_flash_client(flash_return)

    await run_veo_for_clip(
        spec,
        brief,
        0,
        client=client,
        flash_client=flash_client,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        poll_interval_seconds=0.001,
        poll_timeout_seconds=10.0,
    )

    expected_sha = hashlib.sha256(flash_return.encode()).hexdigest()
    pv = video_state.prompt_versions["veo"]
    assert pv.content_sha256 == expected_sha, (
        f"Expected SHA of the rewritten prompt but got SHA of a different string. "
        f"Expected: {expected_sha}, Got: {pv.content_sha256}"
    )
    assert pv.step_name == "veo_generation"
    assert pv.version == veo_prompt.VERSION


# ---------------------------------------------------------------------------
# Test 19 — Safety retry sends full composite (SCENE+AUDIO) to Flash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_safety_retry_sends_full_composite_to_flash(tmp_path):
    """Flash must receive the full composite prompt containing BOTH SCENE: and AUDIO: markers."""
    video_id = str(uuid.uuid4())
    # Use speaking clip (clip 1) so the original prompt has both SCENE and AUDIO
    spec = _make_spec(video_id, clip_count=2)
    brief = _make_brief()
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    (state_root / "videos").mkdir(parents=True)
    (state_root / "runs").mkdir(parents=True)

    firstframe_dir = artifacts_root / video_id
    firstframe_dir.mkdir(parents=True)
    firstframe_path = firstframe_dir / "clip_1_firstframe.png"
    firstframe_path.write_bytes(b"PNG_BYTES")
    video_state.artifacts["clip_1_firstframe"] = str(firstframe_path)

    submit_counter = {"count": 0}
    flash_inputs: list[str] = []

    async def _submit(**kwargs):
        submit_counter["count"] += 1
        return f"op-{submit_counter['count']:03d}"

    async def _poll(op_id):
        if op_id == "op-001":
            return {"done": True, "mp4_bytes": None, "error": None, "safety_block": True}
        return {"done": True, "mp4_bytes": b"RETRY_MP4", "error": None, "safety_block": False}

    client = MagicMock()
    client.submit = AsyncMock(side_effect=_submit)
    client.poll = AsyncMock(side_effect=_poll)

    flash_return = "SCENE: Safe scene.\n\nAUDIO: Testo italiano sicuro."

    async def _capture_rewrite(prompt: str) -> str:
        flash_inputs.append(prompt)
        return flash_return

    flash_client = MagicMock()
    flash_client.rewrite = AsyncMock(side_effect=_capture_rewrite)

    await run_veo_for_clip(
        spec,
        brief,
        1,  # speaking clip — guarantees AUDIO section with Italian script
        client=client,
        flash_client=flash_client,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        poll_interval_seconds=0.001,
        poll_timeout_seconds=10.0,
    )

    assert len(flash_inputs) == 1
    captured = flash_inputs[0]
    # The input to Flash (the safety_retry rendered prompt) must contain both markers
    assert "SCENE:" in captured, "SCENE: marker missing from Flash input"
    assert "AUDIO:" in captured, "AUDIO: marker missing from Flash input"


# ---------------------------------------------------------------------------
# Test 20 — Safety retry: Flash empty response falls back and warns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_safety_retry_flash_empty_response_falls_back_and_warns(tmp_path):
    """When Flash returns an empty (or too-short) string, fall back to the original
    veo_full_prompt and emit a warning event 'safety_retry_flash_returned_empty'."""
    from unittest.mock import patch

    video_id = str(uuid.uuid4())
    spec = _make_spec(video_id, clip_count=2)
    brief = _make_brief()
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    (state_root / "videos").mkdir(parents=True)
    (state_root / "runs").mkdir(parents=True)

    firstframe_dir = artifacts_root / video_id
    firstframe_dir.mkdir(parents=True)
    firstframe_path = firstframe_dir / "clip_0_firstframe.png"
    firstframe_path.write_bytes(b"PNG_BYTES")
    video_state.artifacts["clip_0_firstframe"] = str(firstframe_path)

    submit_counter = {"count": 0}
    submitted_prompts: list[str] = []

    async def _submit(**kwargs):
        submit_counter["count"] += 1
        submitted_prompts.append(kwargs["prompt"])
        return f"op-{submit_counter['count']:03d}"

    async def _poll(op_id):
        if op_id == "op-001":
            return {"done": True, "mp4_bytes": None, "error": None, "safety_block": True}
        return {"done": True, "mp4_bytes": b"RETRY_MP4", "error": None, "safety_block": False}

    client = MagicMock()
    client.submit = AsyncMock(side_effect=_submit)
    client.poll = AsyncMock(side_effect=_poll)

    # Flash returns an empty string (triggers fallback)
    flash_client = _make_mock_flash_client("")

    # Patch the module-level structlog logger to capture warning calls
    warning_calls: list[str] = []
    mock_log = MagicMock()

    def _capture_warning(event: str, **kwargs) -> None:
        warning_calls.append(event)

    mock_log.warning = MagicMock(side_effect=_capture_warning)
    mock_log.info = MagicMock()

    with patch("ugc_pipeline.steps.veo.log", mock_log):
        await run_veo_for_clip(
            spec,
            brief,
            0,
            client=client,
            flash_client=flash_client,
            video_state=video_state,
            run_state=run_state,
            state_root=state_root,
            artifacts_root=artifacts_root,
            poll_interval_seconds=0.001,
            poll_timeout_seconds=10.0,
        )

    assert len(submitted_prompts) == 2

    # The original veo_full_prompt for clip 0 (silent, non-speaking)
    from ugc_pipeline.prompts import veo as veo_prompt_module
    expected_fallback = veo_prompt_module.render(
        scene_description=spec.scene_descriptions[0],
        tone=spec.tone,
        is_speaking_clip=False,
        script_block="",
    )
    # (a) Second submit must use the original prompt as fallback
    assert submitted_prompts[1] == expected_fallback

    # (b) Warning event must have been logged
    assert "safety_retry_flash_returned_empty" in warning_calls, (
        "Expected a warning log event 'safety_retry_flash_returned_empty' but none was found. "
        f"Warning calls: {warning_calls}"
    )

    # (c) prompt_versions["veo"] must NOT have been overwritten — its SHA must still
    # match the original (non-rewritten) prompt, not any Flash output.
    import hashlib
    original_sha = hashlib.sha256(expected_fallback.encode()).hexdigest()
    pv = video_state.prompt_versions.get("veo")
    assert pv is not None, "prompt_versions['veo'] was removed entirely"
    assert pv.content_sha256 == original_sha, (
        "prompt_versions['veo'] SHA was overwritten even though Flash used fallback. "
        f"Expected SHA of original prompt ({original_sha}), got {pv.content_sha256}"
    )


# ---------------------------------------------------------------------------
# Test 20b — Safety retry: Flash response missing structure markers triggers fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_safety_retry_flash_structure_lost_falls_back(tmp_path):
    """When Flash returns a long string that lacks SCENE:/AUDIO: markers, fall back to
    the original veo_full_prompt, log 'safety_retry_flash_structure_lost', and leave
    prompt_versions['veo'] unchanged."""
    import hashlib
    from unittest.mock import patch

    video_id = str(uuid.uuid4())
    spec = _make_spec(video_id, clip_count=2)
    brief = _make_brief()
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    (state_root / "videos").mkdir(parents=True)
    (state_root / "runs").mkdir(parents=True)

    firstframe_dir = artifacts_root / video_id
    firstframe_dir.mkdir(parents=True)
    firstframe_path = firstframe_dir / "clip_0_firstframe.png"
    firstframe_path.write_bytes(b"PNG_BYTES")
    video_state.artifacts["clip_0_firstframe"] = str(firstframe_path)

    submit_counter = {"count": 0}
    submitted_prompts: list[str] = []

    async def _submit(**kwargs):
        submit_counter["count"] += 1
        submitted_prompts.append(kwargs["prompt"])
        return f"op-{submit_counter['count']:03d}"

    async def _poll(op_id):
        if op_id == "op-001":
            return {"done": True, "mp4_bytes": None, "error": None, "safety_block": True}
        return {"done": True, "mp4_bytes": b"RETRY_MP4", "error": None, "safety_block": False}

    client = MagicMock()
    client.submit = AsyncMock(side_effect=_submit)
    client.poll = AsyncMock(side_effect=_poll)

    # Flash returns a long string but with no SCENE:/AUDIO: structural markers
    unstructured_response = "Lorem ipsum " * 50  # 600 chars, no markers
    flash_client = _make_mock_flash_client(unstructured_response)

    warning_calls: list[str] = []
    mock_log = MagicMock()

    def _capture_warning(event: str, **kwargs) -> None:
        warning_calls.append(event)

    mock_log.warning = MagicMock(side_effect=_capture_warning)
    mock_log.info = MagicMock()

    with patch("ugc_pipeline.steps.veo.log", mock_log):
        await run_veo_for_clip(
            spec,
            brief,
            0,
            client=client,
            flash_client=flash_client,
            video_state=video_state,
            run_state=run_state,
            state_root=state_root,
            artifacts_root=artifacts_root,
            poll_interval_seconds=0.001,
            poll_timeout_seconds=10.0,
        )

    # (a) Second submit must equal the original full prompt (fallback)
    from ugc_pipeline.prompts import veo as veo_prompt_module
    expected_fallback = veo_prompt_module.render(
        scene_description=spec.scene_descriptions[0],
        tone=spec.tone,
        is_speaking_clip=False,
        script_block="",
    )
    assert len(submitted_prompts) == 2
    assert submitted_prompts[1] == expected_fallback, (
        "Second submit did not use the original prompt as fallback when Flash lost structure"
    )

    # (b) Structural warning event must have been logged
    assert "safety_retry_flash_structure_lost" in warning_calls, (
        "Expected warning 'safety_retry_flash_structure_lost' but got: "
        f"{warning_calls}"
    )

    # (c) prompt_versions["veo"] must NOT have been overwritten
    original_sha = hashlib.sha256(expected_fallback.encode()).hexdigest()
    pv = video_state.prompt_versions.get("veo")
    assert pv is not None, "prompt_versions['veo'] was removed entirely"
    assert pv.content_sha256 == original_sha, (
        "prompt_versions['veo'] SHA was overwritten even though Flash output lacked structure. "
        f"Expected SHA of original prompt ({original_sha}), got {pv.content_sha256}"
    )


# ---------------------------------------------------------------------------
# Test 16 — _DefaultVeoClient passes generate_audio and does NOT set enhance_prompt
# ---------------------------------------------------------------------------


def test_default_veo_client_passes_generate_audio_to_config():
    """GenerateVideosConfig must receive generate_audio=True and must NOT set enhance_prompt.

    Veo 3.x rejects enhance_prompt=False with INVALID_ARGUMENT (code 3).
    """
    from unittest.mock import MagicMock, patch

    captured_configs: list = []

    def _fake_config(**kwargs):
        captured_configs.append(kwargs)
        return MagicMock()

    with patch("google.genai.types.GenerateVideosConfig", side_effect=_fake_config):
        # We only need to test the __init__ path for the config construction;
        # the google.genai client itself will fail without real credentials,
        # so we patch at the types level and also stub the client constructor.
        with patch("google.genai.Client") as mock_genai_client, \
             patch("google.oauth2.service_account.Credentials.from_service_account_file",
                   return_value=MagicMock()):
            client = _DefaultVeoClient(
                project="test-project",
                location="us-central1",
                credentials_path="/fake/creds.json",
                generate_audio=True,
            )

            # Now call submit to trigger GenerateVideosConfig construction
            import asyncio

            async def _run():
                with patch("google.genai.types.Image", return_value=MagicMock()):
                    mock_genai_client.return_value.aio.models.generate_videos = AsyncMock(
                        return_value=MagicMock(name="op-test-001")
                    )
                    client._client = mock_genai_client.return_value
                    try:
                        await client.submit(image_bytes=b"PNG", prompt="test")
                    except Exception:
                        pass  # op_name extraction may fail; we only care about config kwargs

            asyncio.run(_run())

    assert len(captured_configs) >= 1
    cfg_kwargs = captured_configs[0]
    assert cfg_kwargs.get("generate_audio") is True
    assert "enhance_prompt" not in cfg_kwargs, (
        "enhance_prompt must not be passed to GenerateVideosConfig — "
        "Veo 3.x rejects enhance_prompt=False with INVALID_ARGUMENT (code 3)."
    )


# ---------------------------------------------------------------------------
# Test 17 — prompt_versions["veo"] is recorded after a successful run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_video_state_prompt_versions_records_veo(tmp_path):
    """After run_veo_for_clip succeeds, prompt_versions['veo'] must be set."""
    video_id = str(uuid.uuid4())
    spec = _make_spec(video_id, clip_count=2)
    brief = _make_brief()
    run_state = _make_run_state()
    video_state = _make_video_state(video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    (state_root / "videos").mkdir(parents=True)
    (state_root / "runs").mkdir(parents=True)

    firstframe_dir = artifacts_root / video_id
    firstframe_dir.mkdir(parents=True)
    firstframe_path = firstframe_dir / "clip_0_firstframe.png"
    firstframe_path.write_bytes(b"PNG_BYTES")
    video_state.artifacts["clip_0_firstframe"] = str(firstframe_path)

    client = _make_mock_veo_client(
        poll_responses=[{"done": True, "mp4_bytes": b"MP4_DATA", "error": None, "safety_block": False}]
    )

    await run_veo_for_clip(
        spec,
        brief,
        0,
        client=client,
        flash_client=None,
        video_state=video_state,
        run_state=run_state,
        state_root=state_root,
        artifacts_root=artifacts_root,
        poll_interval_seconds=0.001,
        poll_timeout_seconds=10.0,
    )

    assert "veo" in video_state.prompt_versions
    pv = video_state.prompt_versions["veo"]
    assert pv.step_name == "veo_generation"
    assert pv.version == veo_prompt.VERSION
