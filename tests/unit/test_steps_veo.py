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
from ugc_pipeline.steps.veo import (
    VeoGenerationError,
    VeoOperation,
    VeoResult,
    VeoSafetyBlockError,
    VeoTimeoutError,
    drain_inflight_veo,
    estimate_safety_retry_rewrite_cost_usd,
    estimate_veo_clip_cost_usd,
    poll_veo_operation,
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
        script_blocks=[f"Blocco {i}: testo italiano." for i in range(clip_count)],
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
    assert video_state.artifacts["clip_0_raw"] == str(out_path)
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

    flash_client = _make_mock_flash_client("Softened safe prompt.")

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

    flash_client = _make_mock_flash_client("Still risky rewrite.")

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
