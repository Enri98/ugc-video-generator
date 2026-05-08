"""Unit tests for ugc_pipeline.cost_tracker (SPEC.md §12, §8).

All tests use in-memory model instances; no disk I/O, no API calls.
"""

from __future__ import annotations

import uuid

import pytest

from ugc_pipeline.cost_tracker import (
    BudgetExceededError,
    KillSwitchFiredError,
    admit_clip_batch,
    check_kill_switch,
    check_video_budget,
    increment_cost,
)
from ugc_pipeline.models import RunState, VideoState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_state(**kw: object) -> RunState:
    defaults: dict = {"run_id": str(uuid.uuid4())}
    defaults.update(kw)
    return RunState(**defaults)  # type: ignore[arg-type]


def _video_state(**kw: object) -> VideoState:
    defaults: dict = {
        "video_id": str(uuid.uuid4()),
        "product_id": "abc123def456",
        "spec_index": 0,
    }
    defaults.update(kw)
    return VideoState(**defaults)  # type: ignore[arg-type]


_CFG = {
    "per_video_max_usd": 8.0,
    "global_max_usd": 50.0,
    "max_cost_per_clip_usd": 0.60,
}


# ---------------------------------------------------------------------------
# increment_cost — basic arithmetic
# ---------------------------------------------------------------------------


def test_increment_cost_updates_video_field() -> None:
    rs = _run_state()
    vs = _video_state()
    increment_cost(rs, vs, "veo_usd", 0.40, global_max_usd=50.0)
    assert abs(vs.costs_usd.veo_usd - 0.40) < 1e-9


def test_increment_cost_updates_cumulative() -> None:
    rs = _run_state()
    vs = _video_state()
    increment_cost(rs, vs, "first_frame_usd", 0.05, global_max_usd=50.0)
    assert abs(rs.cumulative_cost_usd - 0.05) < 1e-9


def test_increment_cost_accumulates_across_calls() -> None:
    rs = _run_state()
    vs = _video_state()
    increment_cost(rs, vs, "veo_usd", 0.30, global_max_usd=50.0)
    increment_cost(rs, vs, "veo_usd", 0.40, global_max_usd=50.0)
    assert abs(vs.costs_usd.veo_usd - 0.70) < 1e-9
    assert abs(rs.cumulative_cost_usd - 0.70) < 1e-9


def test_increment_cost_works_without_video_state() -> None:
    rs = _run_state()
    increment_cost(rs, None, "veo_usd", 0.25, global_max_usd=50.0)
    assert abs(rs.cumulative_cost_usd - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# Kill-switch fires at threshold  (SPEC.md §13 example)
# ---------------------------------------------------------------------------


def test_kill_switch_fires_at_threshold() -> None:
    """cumulative=49.95 + 0.10 → 50.05 >= 50.00 → kill_switch_fired=True."""
    rs = _run_state(cumulative_cost_usd=49.95)
    vs = _video_state()
    increment_cost(rs, vs, "veo_usd", 0.10, global_max_usd=50.0)
    assert rs.kill_switch_fired is True


def test_kill_switch_does_not_fire_below_threshold() -> None:
    rs = _run_state(cumulative_cost_usd=49.90)
    vs = _video_state()
    increment_cost(rs, vs, "veo_usd", 0.05, global_max_usd=50.0)
    # 49.90 + 0.05 = 49.95 < 50.00
    assert rs.kill_switch_fired is False


def test_kill_switch_fires_exactly_at_cap() -> None:
    """cumulative=49.00 + 1.00 = 50.00 → fires (>= not >)."""
    rs = _run_state(cumulative_cost_usd=49.00)
    vs = _video_state()
    increment_cost(rs, vs, "veo_usd", 1.00, global_max_usd=50.0)
    assert rs.kill_switch_fired is True


# ---------------------------------------------------------------------------
# check_kill_switch — refuses new work after firing
# ---------------------------------------------------------------------------


def test_check_kill_switch_raises_when_fired() -> None:
    rs = _run_state(kill_switch_fired=True)
    with pytest.raises(KillSwitchFiredError):
        check_kill_switch(rs)


def test_check_kill_switch_passes_when_not_fired() -> None:
    rs = _run_state(kill_switch_fired=False)
    check_kill_switch(rs)  # should not raise


# ---------------------------------------------------------------------------
# check_video_budget
# ---------------------------------------------------------------------------


def test_check_video_budget_raises_when_at_cap() -> None:
    vs = _video_state()
    vs.costs_usd.veo_usd = 8.00  # total == cap
    with pytest.raises(BudgetExceededError):
        check_video_budget(vs, per_video_max_usd=8.00)


def test_check_video_budget_raises_when_over_cap() -> None:
    vs = _video_state()
    vs.costs_usd.veo_usd = 8.50
    with pytest.raises(BudgetExceededError):
        check_video_budget(vs, per_video_max_usd=8.00)


def test_check_video_budget_passes_when_under_cap() -> None:
    vs = _video_state()
    vs.costs_usd.veo_usd = 7.99
    check_video_budget(vs, per_video_max_usd=8.00)  # should not raise


# ---------------------------------------------------------------------------
# admit_clip_batch — per-video trimming
# ---------------------------------------------------------------------------


def test_admit_clip_batch_passes_when_within_budget() -> None:
    rs = _run_state()
    vs = _video_state()
    # 3 clips × $0.60 = $1.80 < $8.00 remaining
    result = admit_clip_batch(vs, [0, 1, 2], rs, _CFG)
    assert result == [0, 1, 2]


def test_admit_clip_batch_trims_when_worst_case_exceeds_per_video() -> None:
    rs = _run_state()
    vs = _video_state()
    vs.costs_usd.veo_usd = 7.00  # remaining = 1.00; 3×0.60=1.80 > 1.00 → trim to 1
    result = admit_clip_batch(vs, [0, 1, 2], rs, _CFG)
    assert result == [0]  # floor(1.00 / 0.60) == 1


def test_admit_clip_batch_raises_when_no_clips_fit_per_video() -> None:
    rs = _run_state()
    vs = _video_state()
    vs.costs_usd.veo_usd = 7.80  # remaining = 0.20 < 0.60 → BudgetExceededError
    with pytest.raises(BudgetExceededError):
        admit_clip_batch(vs, [0], rs, _CFG)


# ---------------------------------------------------------------------------
# admit_clip_batch — global budget check
# ---------------------------------------------------------------------------


def test_admit_clip_batch_raises_when_global_cap_exceeded() -> None:
    rs = _run_state(cumulative_cost_usd=49.60)  # remaining = 0.40
    vs = _video_state()
    # 1 clip × $0.60 = $0.60 > $0.40 remaining globally → BudgetExceededError
    with pytest.raises(BudgetExceededError):
        admit_clip_batch(vs, [0], rs, _CFG)


def test_admit_clip_batch_global_passes_when_within_budget() -> None:
    rs = _run_state(cumulative_cost_usd=48.00)  # remaining = 2.00
    vs = _video_state()
    # 3 × 0.60 = 1.80 < 2.00 — global passes; per-video also fine
    result = admit_clip_batch(vs, [0, 1, 2], rs, _CFG)
    assert result == [0, 1, 2]
