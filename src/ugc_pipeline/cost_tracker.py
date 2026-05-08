"""Cost tracking, budget enforcement, and kill-switch helpers.

Implements the live cost tracking protocol from SPEC.md §12 and the budget
admission control from SPEC.md §8.
"""

from __future__ import annotations

import pathlib
from typing import Any, Coroutine

import structlog

from ugc_pipeline.models import RunState, VideoState
from ugc_pipeline.state_manager import save_run_state, save_video_state

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions  (SPEC.md §11)
# ---------------------------------------------------------------------------


class BudgetExceededError(Exception):
    """Raised when a per-video or global budget cap would be exceeded."""


class KillSwitchFiredError(Exception):
    """Raised when `RunState.kill_switch_fired` is True and new work is attempted."""


# ---------------------------------------------------------------------------
# Core cost mutation
# ---------------------------------------------------------------------------


def increment_cost(
    run_state: RunState,
    video_state: VideoState | None,
    field: str,
    amount_usd: float,
    global_max_usd: float,
) -> None:
    """Increment cost in *video_state* and *run_state*, firing the kill-switch if needed.

    Mutates `video_state.costs_usd.<field>` (if *video_state* is not None) and
    `run_state.cumulative_cost_usd`. Sets `run_state.kill_switch_fired = True` when
    the cumulative total meets or exceeds *global_max_usd*.

    The field name must match a writable attribute of `CostBreakdown` (e.g.
    `"veo_usd"`, `"first_frame_usd"`).
    """
    if video_state is not None:
        current = getattr(video_state.costs_usd, field)
        setattr(video_state.costs_usd, field, current + amount_usd)

    run_state.cumulative_cost_usd += amount_usd

    if run_state.cumulative_cost_usd >= global_max_usd:
        run_state.kill_switch_fired = True


# ---------------------------------------------------------------------------
# Budget guard helpers
# ---------------------------------------------------------------------------


def check_video_budget(video_state: VideoState, per_video_max_usd: float) -> None:
    """Raise BudgetExceededError if this video's total cost meets or exceeds the cap."""
    if video_state.costs_usd.total >= per_video_max_usd:
        raise BudgetExceededError(
            f"Video {video_state.video_id!r} has reached the per-video budget cap "
            f"(${per_video_max_usd:.2f}). Current total: ${video_state.costs_usd.total:.4f}."
        )


def check_kill_switch(run_state: RunState) -> None:
    """Raise KillSwitchFiredError if the global kill-switch has been activated."""
    if run_state.kill_switch_fired:
        raise KillSwitchFiredError(
            f"Global budget cap reached. "
            f"Cumulative cost: ${run_state.cumulative_cost_usd:.4f}. "
            "No new work will be admitted."
        )


# ---------------------------------------------------------------------------
# Live cost tracking wrapper  (SPEC.md §12 "Live Cost Tracking Protocol")
# ---------------------------------------------------------------------------


async def track_cost_and_call(
    state: VideoState,
    run_state: RunState,
    field: str,
    cost_usd: float,
    api_coro: Coroutine[Any, Any, Any],
    global_max_usd: float,
    video_state_path: pathlib.Path,
    run_state_path: pathlib.Path,
) -> Any:
    """Record cost atomically, then await the API coroutine.

    Cost is persisted to disk BEFORE the API call so that a crash during the call
    does not lose the cost from accounting on the next run (SPEC.md §12).

    Returns whatever *api_coro* returns.
    """
    # Increment in memory
    increment_cost(run_state, state, field, cost_usd, global_max_usd)

    # Persist both states atomically before the paid call
    from ugc_pipeline.state_manager import _dump_model, write_state_atomic  # local import to avoid circularity

    write_state_atomic(video_state_path, _dump_model(state))
    write_state_atomic(run_state_path, _dump_model(run_state))

    # Now make the actual API call
    return await api_coro


# ---------------------------------------------------------------------------
# Budget admission control  (SPEC.md §8)
# ---------------------------------------------------------------------------


def admit_clip_batch(
    video_state: VideoState,
    clip_indices: list[int],
    run_state: RunState,
    cfg: dict[str, Any],
) -> list[int]:
    """Return a (possibly trimmed) list of clip indices that fit within budget.

    *cfg* must contain keys: `budget.per_video_max_usd`, `budget.global_max_usd`,
    `budget.max_cost_per_clip_usd`.

    Raises `BudgetExceededError` when:
    - No clips fit within the remaining per-video budget.
    - The full (or trimmed) batch would exceed the remaining global budget.
    """
    budget_cfg = cfg.get("budget", cfg)  # accept flat or nested dict
    max_cost_per_clip: float = float(budget_cfg["max_cost_per_clip_usd"])
    per_video_max: float = float(budget_cfg["per_video_max_usd"])
    global_max: float = float(budget_cfg["global_max_usd"])

    worst_case = len(clip_indices) * max_cost_per_clip

    # Per-video check
    remaining_video = per_video_max - video_state.costs_usd.total
    if worst_case > remaining_video:
        affordable = int(remaining_video // max_cost_per_clip)
        if affordable == 0:
            raise BudgetExceededError(
                f"Video {video_state.video_id!r} has no remaining per-video budget "
                f"(remaining: ${remaining_video:.4f}, min clip cost: ${max_cost_per_clip:.4f})."
            )
        log.warning(
            "budget_admission_trimmed_batch",
            video_id=video_state.video_id,
            original=len(clip_indices),
            admitted=affordable,
            remaining_video_usd=remaining_video,
        )
        clip_indices = clip_indices[:affordable]

    # Global check — re-compute worst_case for the trimmed batch
    worst_case_trimmed = len(clip_indices) * max_cost_per_clip
    remaining_global = global_max - run_state.cumulative_cost_usd
    if worst_case_trimmed > remaining_global:
        raise BudgetExceededError(
            f"Global budget cap would be exceeded by this clip batch "
            f"(worst-case: ${worst_case_trimmed:.4f}, remaining: ${remaining_global:.4f})."
        )

    return clip_indices
