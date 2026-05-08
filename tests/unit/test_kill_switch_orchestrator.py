"""Kill-switch orchestrator tests — SPEC.md §12 and §13.

Covers:
  1. Kill-switch fires deterministically when cumulative_cost >= global_max.
  2. admit_product refuses new work when kill_switch_fired is True.
  3. drain_inflight_veo polls to completion (does not abandon operations).
"""

from __future__ import annotations

import asyncio
import pathlib
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from ugc_pipeline.cost_tracker import KillSwitchFiredError, increment_cost
from ugc_pipeline.models import RunState, VideoState
from ugc_pipeline.orchestrator import OrchestratorContext, admit_product
from ugc_pipeline.steps.veo import (
    VeoOperation,
    VeoResult,
    drain_inflight_veo,
    VeoSafetyBlockError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_state(**kw: object) -> RunState:
    return RunState(run_id=str(uuid.uuid4()), **kw)  # type: ignore[arg-type]


def _video_state(video_id: str | None = None) -> VideoState:
    return VideoState(
        video_id=video_id or str(uuid.uuid4()),
        product_id="abc123def456",
        spec_index=0,
    )


# ---------------------------------------------------------------------------
# Test 1: Kill-switch fires at threshold
# ---------------------------------------------------------------------------


def test_kill_switch_fires_at_global_max() -> None:
    """cumulative_cost >= global_max must flip kill_switch_fired to True."""
    run_state = _run_state()
    run_state.cumulative_cost_usd = 49.95
    global_max = 50.0

    # This increment pushes it to 50.05
    increment_cost(run_state, None, "veo_usd", 0.10, global_max_usd=global_max)

    assert run_state.cumulative_cost_usd >= global_max
    assert run_state.kill_switch_fired is True


def test_kill_switch_fires_exactly_at_max() -> None:
    """cumulative_cost == global_max (not just >) must still flip the switch."""
    run_state = _run_state()
    run_state.cumulative_cost_usd = 49.90
    global_max = 50.0

    increment_cost(run_state, None, "veo_usd", 0.10, global_max_usd=global_max)

    assert run_state.kill_switch_fired is True


def test_kill_switch_does_not_fire_below_max() -> None:
    """cumulative_cost < global_max must NOT flip the switch."""
    run_state = _run_state()
    run_state.cumulative_cost_usd = 49.0
    global_max = 50.0

    increment_cost(run_state, None, "veo_usd", 0.50, global_max_usd=global_max)

    assert run_state.cumulative_cost_usd < global_max
    assert run_state.kill_switch_fired is False


# ---------------------------------------------------------------------------
# Test 2: Orchestrator refuses new work after kill-switch fires
# ---------------------------------------------------------------------------


def test_admit_product_refuses_when_kill_switch_fired() -> None:
    """admit_product must raise KillSwitchFiredError when kill_switch_fired is True."""
    run_state = _run_state(kill_switch_fired=True)
    with pytest.raises(KillSwitchFiredError):
        admit_product(run_state, "product-abc123")


def test_admit_product_passes_when_not_fired() -> None:
    """admit_product must not raise when kill_switch_fired is False."""
    run_state = _run_state(kill_switch_fired=False)
    admit_product(run_state, "product-abc123")  # must not raise


# ---------------------------------------------------------------------------
# Test 3: drain_inflight_veo polls to completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_inflight_veo_polls_until_done(tmp_path: pathlib.Path) -> None:
    """drain_inflight_veo must poll each operation until done, not abandon it."""
    mp4_bytes = b"fakemp4" + b"\x00" * 50
    call_count = {"n": 0}

    async def _poll(operation_id: str) -> dict:
        call_count["n"] += 1
        if call_count["n"] < 2:
            return {"done": False}  # first call: still pending
        return {"done": True, "mp4_bytes": mp4_bytes, "error": None, "safety_block": False}

    mock_veo_client = MagicMock()
    mock_veo_client.poll = AsyncMock(side_effect=_poll)

    video_id = str(uuid.uuid4())
    op = VeoOperation(operation_id="drain-op-001", clip_index=0, video_id=video_id)

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    state_root.mkdir()
    artifacts_root.mkdir()

    run_state = _run_state(kill_switch_fired=True)
    vs = _video_state(video_id=video_id)

    await drain_inflight_veo(
        operations=[op],
        client=mock_veo_client,
        run_state=run_state,
        video_states_by_video_id={video_id: vs},
        state_root=state_root,
        artifacts_root=artifacts_root,
        poll_interval_seconds=0.001,
        poll_timeout_seconds=10.0,
    )

    # Must have polled at least 2 times (pending → done)
    assert call_count["n"] >= 2, f"Expected >= 2 poll calls, got {call_count['n']}"

    # Artifact must have been written to disk
    expected_path = artifacts_root / video_id / "clip_0_raw.mp4"
    assert expected_path.exists(), f"Expected artifact at {expected_path}"
    assert expected_path.read_bytes() == mp4_bytes


@pytest.mark.asyncio
async def test_drain_inflight_veo_handles_safety_block(tmp_path: pathlib.Path) -> None:
    """drain_inflight_veo must continue draining other ops even if one hits a safety block."""
    async def _poll_blocked(operation_id: str) -> dict:
        return {"done": True, "mp4_bytes": None, "error": None, "safety_block": True}

    mp4_bytes = b"fakemp4good"

    async def _poll_ok(operation_id: str) -> dict:
        return {"done": True, "mp4_bytes": mp4_bytes, "error": None, "safety_block": False}

    call_order: list[str] = []

    async def _dispatch_poll(operation_id: str) -> dict:
        call_order.append(operation_id)
        if operation_id == "op-blocked":
            return await _poll_blocked(operation_id)
        return await _poll_ok(operation_id)

    mock_client = MagicMock()
    mock_client.poll = AsyncMock(side_effect=_dispatch_poll)

    vid1 = str(uuid.uuid4())
    vid2 = str(uuid.uuid4())
    ops = [
        VeoOperation(operation_id="op-blocked", clip_index=0, video_id=vid1),
        VeoOperation(operation_id="op-ok", clip_index=0, video_id=vid2),
    ]

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    state_root.mkdir()
    artifacts_root.mkdir()

    run_state = _run_state(kill_switch_fired=True)
    vs1 = _video_state(video_id=vid1)
    vs2 = _video_state(video_id=vid2)

    await drain_inflight_veo(
        operations=ops,
        client=mock_client,
        run_state=run_state,
        video_states_by_video_id={vid1: vs1, vid2: vs2},
        state_root=state_root,
        artifacts_root=artifacts_root,
        poll_interval_seconds=0.001,
        poll_timeout_seconds=5.0,
    )

    # Both operations must have been polled
    assert "op-blocked" in call_order
    assert "op-ok" in call_order

    # Only the OK op produces an artifact
    ok_artifact = artifacts_root / vid2 / "clip_0_raw.mp4"
    assert ok_artifact.exists()
