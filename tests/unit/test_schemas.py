"""Unit tests for Pydantic models in ugc_pipeline.models (SPEC.md §4).

Validates required fields, defaults, computed properties, and invalid input rejection.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from ugc_pipeline.models import (
    CostBreakdown,
    ProductBrief,
    PromptVersion,
    Report,
    RunState,
    VideoSpec,
    VideoState,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_brief(**overrides: object) -> ProductBrief:
    defaults: dict = {
        "product_id": "a3f9c12e7b04",
        "image_path": "artifacts/a3f9c12e7b04/product.jpg",
        "shape": "cylindrical mug with a C-shaped handle",
        "dominant_colours": ["#F5F0E8", "#3B2A1A"],
        "packaging_style": "kraft paper box",
        "inferred_category": "kitchenware",
        "lifestyle_contexts": ["morning routine", "desk setup"],
    }
    defaults.update(overrides)
    return ProductBrief(**defaults)  # type: ignore[arg-type]


def _make_spec(**overrides: object) -> VideoSpec:
    defaults: dict = {
        "video_id": str(uuid.uuid4()),
        "product_id": "a3f9c12e7b04",
        "spec_index": 0,
        "tone": "warm storyteller",
        "narrative_arc": "A quiet morning transforms into comfort.",
        "talent_id": "talent_02",
        "clip_count": 2,
        "scene_descriptions": ["Scene A.", "Scene B."],
        "script_blocks": ["", "Testo italiano due."],
    }
    defaults.update(overrides)
    return VideoSpec(**defaults)  # type: ignore[arg-type]


def _make_video_state(**overrides: object) -> VideoState:
    defaults: dict = {
        "video_id": str(uuid.uuid4()),
        "product_id": "a3f9c12e7b04",
        "spec_index": 0,
    }
    defaults.update(overrides)
    return VideoState(**defaults)  # type: ignore[arg-type]


def _make_run_state(**overrides: object) -> RunState:
    defaults: dict = {"run_id": str(uuid.uuid4())}
    defaults.update(overrides)
    return RunState(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CostBreakdown
# ---------------------------------------------------------------------------


def test_cost_breakdown_defaults_to_zero() -> None:
    cost = CostBreakdown()
    assert cost.product_analyst_usd == 0.0
    assert cost.creative_director_usd == 0.0
    assert cost.first_frame_usd == 0.0
    assert cost.veo_usd == 0.0
    assert cost.safety_retry_usd == 0.0
    assert cost.caption_usd == 0.0


def test_cost_breakdown_total_is_sum() -> None:
    cost = CostBreakdown(
        product_analyst_usd=0.01,
        creative_director_usd=0.02,
        first_frame_usd=0.05,
        veo_usd=0.40,
        safety_retry_usd=0.001,
        caption_usd=0.0,
    )
    expected = 0.01 + 0.02 + 0.05 + 0.40 + 0.001
    assert abs(cost.total - expected) < 1e-9


def test_cost_breakdown_total_updates_on_mutation() -> None:
    cost = CostBreakdown(veo_usd=0.30)
    assert abs(cost.total - 0.30) < 1e-9
    cost.veo_usd += 0.20
    assert abs(cost.total - 0.50) < 1e-9


def test_cost_breakdown_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CostBreakdown(unknown_field=99.0)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ProductBrief
# ---------------------------------------------------------------------------


def test_product_brief_required_fields() -> None:
    brief = _make_brief()
    assert brief.product_id == "a3f9c12e7b04"
    assert brief.visual_notes is None  # optional, defaults to None


def test_product_brief_created_at_is_utc() -> None:
    brief = _make_brief()
    # The default factory uses datetime.now(timezone.utc), so tzinfo must be set
    assert brief.created_at.tzinfo is not None


def test_product_brief_rejects_missing_required() -> None:
    with pytest.raises(ValidationError):
        ProductBrief(product_id="abc")  # type: ignore[call-arg]  # missing many fields


def test_product_brief_accepts_visual_notes() -> None:
    brief = _make_brief(visual_notes="Matte finish with subtle texture.")
    assert brief.visual_notes == "Matte finish with subtle texture."


# ---------------------------------------------------------------------------
# VideoSpec
# ---------------------------------------------------------------------------


def test_video_spec_required_fields() -> None:
    spec = _make_spec()
    assert spec.spec_index == 0
    assert spec.clip_count == 2
    assert spec.visual_style_notes is None


def test_video_spec_created_at_is_utc() -> None:
    spec = _make_spec()
    assert spec.created_at.tzinfo is not None


# ---------------------------------------------------------------------------
# VideoState
# ---------------------------------------------------------------------------


def test_video_state_completed_steps_defaults_to_empty_list() -> None:
    state = _make_video_state()
    assert state.completed_steps == []


def test_video_state_artifacts_defaults_to_empty_dict() -> None:
    state = _make_video_state()
    assert state.artifacts == {}


def test_video_state_costs_usd_defaults_to_zero_breakdown() -> None:
    state = _make_video_state()
    assert state.costs_usd.total == 0.0


def test_video_state_status_defaults_to_pending() -> None:
    state = _make_video_state()
    assert state.status == "pending"


def test_video_state_accepts_valid_status_literals() -> None:
    valid_statuses = [
        "pending",
        "in_progress",
        "completed",
        "failed_safety",
        "failed_budget",
        "failed_error",
        "failed_timeout",
    ]
    for status in valid_statuses:
        state = _make_video_state(status=status)
        assert state.status == status


def test_video_state_rejects_invalid_status() -> None:
    with pytest.raises(ValidationError):
        _make_video_state(status="unknown_status")


def test_video_state_prompt_versions_defaults_to_empty_dict() -> None:
    state = _make_video_state()
    assert state.prompt_versions == {}


# ---------------------------------------------------------------------------
# RunState
# ---------------------------------------------------------------------------


def test_run_state_defaults() -> None:
    rs = _make_run_state()
    assert rs.cumulative_cost_usd == 0.0
    assert rs.kill_switch_fired is False
    assert rs.products_seen == []
    assert rs.videos_completed == 0
    assert rs.videos_failed == 0
    assert rs.ended_at is None


def test_run_state_started_at_is_utc() -> None:
    rs = _make_run_state()
    assert rs.started_at.tzinfo is not None


# ---------------------------------------------------------------------------
# PromptVersion
# ---------------------------------------------------------------------------


def test_prompt_version_fields() -> None:
    pv = PromptVersion(
        step_name="product_analyst",
        version="1.0.0",
        content_sha256="abc123" * 10,
    )
    assert pv.step_name == "product_analyst"
    assert pv.version == "1.0.0"
    assert pv.rendered_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_required_fields() -> None:
    report = Report(
        run_id=str(uuid.uuid4()),
        product_id="a3f9c12e7b04",
        video_id=str(uuid.uuid4()),
        talent_id="talent_02",
        spec_index=0,
        scene_descriptions=["Scene A."],
        script_blocks=["Testo italiano."],
        token_counts={"product_analyst": 1500},
        costs_usd=CostBreakdown(veo_usd=0.40),
        status="completed",
    )
    assert report.status == "completed"
    assert report.costs_usd.total == 0.40
