"""Unit tests for src/ugc_pipeline/steps/report_gen.py.

Covers:
- render_report_text produces all expected sections.
- Costs formatted to 4 decimal places.
- Italian script_blocks preserved verbatim.
- dry_run=True prepends [DRY RUN] header.
- run_report_gen writes the file and is idempotent.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ugc_pipeline.models import (
    CostBreakdown,
    Report,
    RunState,
    VideoSpec,
    VideoState,
)
from ugc_pipeline.steps.report_gen import render_report_text, run_report_gen


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_costs(**overrides: float) -> CostBreakdown:
    defaults: dict[str, float] = {
        "product_analyst_usd": 0.0123,
        "creative_director_usd": 0.0456,
        "first_frame_usd": 0.0789,
        "veo_usd": 1.2345,
        "safety_retry_usd": 0.0010,
        "caption_usd": 0.0000,
    }
    defaults.update(overrides)
    return CostBreakdown(**defaults)


def _make_video_spec(
    video_id: str,
    product_id: str,
    *,
    scene_descriptions: list[str] | None = None,
    script_blocks: list[str] | None = None,
) -> VideoSpec:
    if scene_descriptions is None:
        scene_descriptions = [
            "Close-up of the ceramic mug on a wooden kitchen counter at dawn",
            "Talent's hands cradle the mug; steam rises into morning light",
        ]
    if script_blocks is None:
        # Italian narrative content (voiceover language convention)
        script_blocks = [
            "",
            "Questo momento è solo tuo.",
        ]
    return VideoSpec(
        video_id=video_id,
        product_id=product_id,
        spec_index=0,
        tone="warm storyteller",
        narrative_arc="A quiet morning routine elevated by a single beautiful object.",
        talent_id="talent_02",
        clip_count=2,
        scene_descriptions=scene_descriptions,
        script_blocks=script_blocks,
    )


def _make_video_state(video_id: str, product_id: str, costs: CostBreakdown) -> VideoState:
    return VideoState(
        video_id=video_id,
        product_id=product_id,
        spec_index=0,
        status="completed",
        costs_usd=costs,
    )


def _make_run_state() -> RunState:
    return RunState(run_id=str(uuid.uuid4()))


def _make_report(
    video_id: str,
    product_id: str,
    run_id: str,
    costs: CostBreakdown,
    spec: VideoSpec,
) -> Report:
    return Report(
        run_id=run_id,
        product_id=product_id,
        video_id=video_id,
        talent_id=spec.talent_id,
        spec_index=spec.spec_index,
        scene_descriptions=spec.scene_descriptions,
        script_blocks=spec.script_blocks,
        token_counts={},
        costs_usd=costs,
        status="completed",
        generated_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def video_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture()
def product_id() -> str:
    return "a3f9c12e7b04"


@pytest.fixture()
def costs() -> CostBreakdown:
    return _make_costs()


@pytest.fixture()
def video_spec(video_id: str, product_id: str) -> VideoSpec:
    return _make_video_spec(video_id, product_id)


@pytest.fixture()
def video_state(video_id: str, product_id: str, costs: CostBreakdown) -> VideoState:
    return _make_video_state(video_id, product_id, costs)


@pytest.fixture()
def run_state() -> RunState:
    return _make_run_state()


@pytest.fixture()
def report(
    video_id: str,
    product_id: str,
    run_state: RunState,
    costs: CostBreakdown,
    video_spec: VideoSpec,
) -> Report:
    return _make_report(video_id, product_id, run_state.run_id, costs, video_spec)


# ---------------------------------------------------------------------------
# render_report_text — structure
# ---------------------------------------------------------------------------


def test_report_contains_header(
    report: Report, video_state: VideoState, video_spec: VideoSpec
) -> None:
    text = render_report_text(report, video_state, video_spec)
    assert "UGC PIPELINE REPORT" in text
    assert "===================" in text


def test_report_contains_run_id(
    report: Report, video_state: VideoState, video_spec: VideoSpec
) -> None:
    text = render_report_text(report, video_state, video_spec)
    assert report.run_id in text


def test_report_contains_product_id(
    report: Report, video_state: VideoState, video_spec: VideoSpec
) -> None:
    text = render_report_text(report, video_state, video_spec)
    assert report.product_id in text


def test_report_contains_video_id(
    report: Report, video_state: VideoState, video_spec: VideoSpec
) -> None:
    text = render_report_text(report, video_state, video_spec)
    assert report.video_id in text


def test_report_contains_talent_id(
    report: Report, video_state: VideoState, video_spec: VideoSpec
) -> None:
    text = render_report_text(report, video_state, video_spec)
    assert report.talent_id in text


def test_report_contains_status(
    report: Report, video_state: VideoState, video_spec: VideoSpec
) -> None:
    text = render_report_text(report, video_state, video_spec)
    assert "completed" in text


def test_report_contains_scenes_section(
    report: Report, video_state: VideoState, video_spec: VideoSpec
) -> None:
    text = render_report_text(report, video_state, video_spec)
    assert "SCENES" in text
    assert "------" in text


def test_report_contains_english_scene_descriptions(
    report: Report, video_state: VideoState, video_spec: VideoSpec
) -> None:
    text = render_report_text(report, video_state, video_spec)
    for desc in video_spec.scene_descriptions:
        assert desc in text


def test_report_preserves_italian_script_blocks(
    report: Report, video_state: VideoState, video_spec: VideoSpec
) -> None:
    """Italian voiceover text must appear verbatim in the report."""
    text = render_report_text(report, video_state, video_spec)
    for block in video_spec.script_blocks:
        assert block in text


def test_report_contains_costs_section(
    report: Report, video_state: VideoState, video_spec: VideoSpec
) -> None:
    text = render_report_text(report, video_state, video_spec)
    assert "COSTS (USD)" in text
    assert "TOTAL:" in text


def test_report_costs_formatted_to_4_decimals(
    report: Report, video_state: VideoState, video_spec: VideoSpec, costs: CostBreakdown
) -> None:
    text = render_report_text(report, video_state, video_spec)
    # product_analyst_usd = 0.0123 → formatted as $0.0123
    assert "$0.0123" in text
    # veo_usd = 1.2345 → formatted as $1.2345
    assert "$1.2345" in text


def test_report_total_cost_in_text(
    report: Report, video_state: VideoState, video_spec: VideoSpec, costs: CostBreakdown
) -> None:
    text = render_report_text(report, video_state, video_spec)
    total_str = f"${costs.total:.4f}"
    assert total_str in text


def test_report_token_counts_section(
    report: Report, video_state: VideoState, video_spec: VideoSpec
) -> None:
    text = render_report_text(report, video_state, video_spec)
    assert "TOKEN COUNTS" in text


def test_report_scene_numbered(
    report: Report, video_state: VideoState, video_spec: VideoSpec
) -> None:
    text = render_report_text(report, video_state, video_spec)
    assert "[1]" in text
    assert "[2]" in text


# ---------------------------------------------------------------------------
# render_report_text — dry_run
# ---------------------------------------------------------------------------


def test_dry_run_prepends_header(
    report: Report, video_state: VideoState, video_spec: VideoSpec
) -> None:
    text = render_report_text(report, video_state, video_spec, dry_run=True)
    assert text.startswith("[DRY RUN]")


def test_dry_run_note_about_estimates(
    report: Report, video_state: VideoState, video_spec: VideoSpec
) -> None:
    text = render_report_text(report, video_state, video_spec, dry_run=True)
    assert "estimates" in text.lower() or "estimate" in text.lower()


def test_non_dry_run_no_dry_run_header(
    report: Report, video_state: VideoState, video_spec: VideoSpec
) -> None:
    text = render_report_text(report, video_state, video_spec, dry_run=False)
    assert "[DRY RUN]" not in text


# ---------------------------------------------------------------------------
# run_report_gen — writes file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_report_gen_writes_file(
    video_state: VideoState,
    video_spec: VideoSpec,
    run_state: RunState,
    tmp_path: Path,
) -> None:
    out = await run_report_gen(
        video_state,
        video_spec,
        run_state,
        artifacts_root=tmp_path,
    )
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.name == "report.txt"


@pytest.mark.asyncio
async def test_run_report_gen_sets_artifact(
    video_state: VideoState,
    video_spec: VideoSpec,
    run_state: RunState,
    tmp_path: Path,
) -> None:
    await run_report_gen(
        video_state,
        video_spec,
        run_state,
        artifacts_root=tmp_path,
    )
    assert "report" in video_state.artifacts
    assert Path(video_state.artifacts["report"]).exists()


@pytest.mark.asyncio
async def test_run_report_gen_content_has_video_id(
    video_state: VideoState,
    video_spec: VideoSpec,
    run_state: RunState,
    tmp_path: Path,
) -> None:
    out = await run_report_gen(
        video_state,
        video_spec,
        run_state,
        artifacts_root=tmp_path,
    )
    content = out.read_text(encoding="utf-8")
    assert video_state.video_id in content


# ---------------------------------------------------------------------------
# run_report_gen — idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_report_gen_idempotent(
    video_state: VideoState,
    video_spec: VideoSpec,
    run_state: RunState,
    tmp_path: Path,
) -> None:
    out1 = await run_report_gen(
        video_state,
        video_spec,
        run_state,
        artifacts_root=tmp_path,
    )
    # Write a sentinel to the file to detect if it gets overwritten
    sentinel = "SENTINEL-DO-NOT-OVERWRITE"
    out1.write_text(sentinel, encoding="utf-8")

    # Second call must skip (file exists with non-zero size)
    out2 = await run_report_gen(
        video_state,
        video_spec,
        run_state,
        artifacts_root=tmp_path,
    )
    assert out1 == out2
    # Content should still be the sentinel (not overwritten)
    assert out2.read_text(encoding="utf-8") == sentinel
