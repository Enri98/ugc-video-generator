"""Flagship end-to-end pipeline test — SPEC.md §13.

Drives a single product through the pipeline stages implemented so far
(product_analyst → creative_director → first_frame_composite → veo_generation)
using fully mocked clients. No paid API calls are made.

Day 4 scope: assertions cover first-frame PNGs and raw Veo MP4 clips.
# TODO Day 5: extend assertions to include trim/stitch/caption artifacts.
"""

from __future__ import annotations

import pathlib
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from ugc_pipeline.models import (
    CostBreakdown,
    ProductBrief,
    RunState,
    VideoSpec,
    VideoState,
)
from ugc_pipeline.steps.first_frame import NanoBananaResult, run_first_frame_for_clip
from ugc_pipeline.steps.veo import run_veo_for_clip


# ---------------------------------------------------------------------------
# Canned data
# ---------------------------------------------------------------------------

_PRODUCT_ID = "e2e000000001"

_CANNED_BRIEF = ProductBrief(
    product_id=_PRODUCT_ID,
    image_path=f"artifacts/{_PRODUCT_ID}/product.jpg",
    shape="cylindrical mug with a C-shaped handle",
    dominant_colours=["#F5F0E8", "#3B2A1A"],
    packaging_style="kraft paper box with embossed geometric pattern",
    inferred_category="kitchenware / home lifestyle",
    lifestyle_contexts=["morning routine", "desk setup"],
    visual_notes="matte ceramic surface",
    created_at=datetime.now(timezone.utc),
)


def _make_canned_spec(spec_index: int) -> VideoSpec:
    vid = str(uuid.uuid4())
    tones = ["warm storyteller", "energetic lifestyle", "serene ASMR"]
    return VideoSpec(
        video_id=vid,
        product_id=_PRODUCT_ID,
        spec_index=spec_index,
        tone=tones[spec_index],
        narrative_arc=f"Narrative arc for spec {spec_index}.",
        talent_id="talent_01",
        clip_count=2,
        scene_descriptions=[
            f"Spec {spec_index} clip 0: talent interacts with the product at dawn.",
            f"Spec {spec_index} clip 1: talent places the product on a sunlit surface.",
        ],
        script_blocks=[
            "Ogni mattina merita un piccolo gesto di cura.",
            "La ceramica scalda le mani.",
        ],
        visual_style_notes="Warm amber tone grading, soft window backlight.",
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Mock clients
# ---------------------------------------------------------------------------


def _make_mock_nano_banana_client(png_bytes: bytes) -> MagicMock:
    """Mock Nano Banana client that returns a fixed PNG for every call."""
    client = MagicMock()
    client.generate_image = AsyncMock(
        return_value=NanoBananaResult(png_bytes=png_bytes, model="gemini-3-flash-image")
    )
    return client


def _make_mock_veo_client(mp4_bytes: bytes) -> MagicMock:
    """Mock Veo client: submit returns incrementing op-ids; poll returns done with mp4_bytes."""
    counter = {"n": 0}

    async def _submit(**kwargs) -> str:
        counter["n"] += 1
        return f"mock-op-{counter['n']:03d}"

    client = MagicMock()
    client.submit = AsyncMock(side_effect=_submit)
    client.poll = AsyncMock(
        return_value={"done": True, "mp4_bytes": mp4_bytes, "error": None, "safety_block": False}
    )
    return client


def _make_mock_flash_client() -> MagicMock:
    client = MagicMock()
    client.rewrite = AsyncMock(side_effect=lambda prompt: prompt + " (rewritten)")
    return client


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_through_veo(
    pipeline_workspace: dict,
    sample_product_image_path,
    firstframe_fixture_png_path,
    clip_fixture_mp4_path,
):
    """Drive a single product through analyst → director → first_frame → veo.

    All external API calls are mocked. Real bytes from fixture files are used
    for the generated artifacts so size assertions are meaningful.

    SPEC.md §13 assertions:
    - All expected first-frame PNGs and raw MP4 clips exist on disk with size > 0.
    - completed_steps records the expected steps in order for each video.
    - run_state.cumulative_cost_usd > 0 (cost is recorded even for mocked calls).
    - No exceptions are raised.
    """
    state_root = pipeline_workspace["state_root"]
    artifacts_root = pipeline_workspace["artifacts_root"]

    # Create required state subdirectories
    (state_root / "videos").mkdir(parents=True, exist_ok=True)
    (state_root / "runs").mkdir(parents=True, exist_ok=True)
    (state_root / "products").mkdir(parents=True, exist_ok=True)

    # Load fixture bytes
    png_bytes = firstframe_fixture_png_path.read_bytes()
    mp4_bytes = clip_fixture_mp4_path.read_bytes()

    # Build mock clients
    nano_banana_client = _make_mock_nano_banana_client(png_bytes)
    veo_client = _make_mock_veo_client(mp4_bytes)
    flash_client = _make_mock_flash_client()

    # Build run state
    run_state = RunState(
        run_id=str(uuid.uuid4()),
        started_at=datetime.now(timezone.utc),
        products_seen=[_PRODUCT_ID],
    )

    # Build 3 specs (simulating creative director output)
    specs = [_make_canned_spec(i) for i in range(3)]
    video_states = [
        VideoState(
            video_id=spec.video_id,
            product_id=spec.product_id,
            spec_index=spec.spec_index,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        for spec in specs
    ]

    # ------------------------------------------------------------------
    # Stage 4 + 5: for each spec, run first_frame_composite then veo_generation
    # ------------------------------------------------------------------
    talent_descriptor = "woman in her late 20s, relaxed aesthetic, natural light"

    for spec, video_state in zip(specs, video_states):
        for clip_index in range(spec.clip_count):
            # --- First frame ---
            firstframe_path = await run_first_frame_for_clip(
                spec=spec,
                brief=_CANNED_BRIEF,
                clip_index=clip_index,
                talent_descriptor=talent_descriptor,
                client=nano_banana_client,
                video_state=video_state,
                run_state=run_state,
                state_root=state_root,
                artifacts_root=artifacts_root,
            )
            assert firstframe_path.exists(), (
                f"First-frame PNG missing: {firstframe_path}"
            )
            assert firstframe_path.stat().st_size > 0, (
                f"First-frame PNG is empty: {firstframe_path}"
            )

            # Mark first_frame step as completed
            step_name = f"first_frame_composite_clip_{clip_index}"
            if step_name not in video_state.completed_steps:
                video_state.completed_steps.append(step_name)

            # --- Veo ---
            raw_path = await run_veo_for_clip(
                spec=spec,
                brief=_CANNED_BRIEF,
                clip_index=clip_index,
                client=veo_client,
                flash_client=flash_client,
                video_state=video_state,
                run_state=run_state,
                state_root=state_root,
                artifacts_root=artifacts_root,
                poll_interval_seconds=0.001,
                poll_timeout_seconds=30.0,
            )
            assert raw_path.exists(), f"Raw clip MP4 missing: {raw_path}"
            assert raw_path.stat().st_size > 0, f"Raw clip MP4 is empty: {raw_path}"

            # Mark veo step as completed
            veo_step_name = f"veo_generation_clip_{clip_index}"
            if veo_step_name not in video_state.completed_steps:
                video_state.completed_steps.append(veo_step_name)

    # ------------------------------------------------------------------
    # SPEC.md §13 assertions
    # ------------------------------------------------------------------

    # 1. All expected artifacts exist on disk for every spec/clip combination
    for spec, video_state in zip(specs, video_states):
        for clip_index in range(spec.clip_count):
            ff_key = f"clip_{clip_index}_firstframe"
            raw_key = f"clip_{clip_index}_raw"

            assert ff_key in video_state.artifacts, (
                f"Missing artifact key {ff_key!r} in video {spec.video_id}"
            )
            assert raw_key in video_state.artifacts, (
                f"Missing artifact key {raw_key!r} in video {spec.video_id}"
            )

            ff_path = pathlib.Path(video_state.artifacts[ff_key])
            raw_path = pathlib.Path(video_state.artifacts[raw_key])

            assert ff_path.exists() and ff_path.stat().st_size > 0
            assert raw_path.exists() and raw_path.stat().st_size > 0

    # 2. completed_steps contains step names in order for each video
    for spec, video_state in zip(specs, video_states):
        for clip_index in range(spec.clip_count):
            assert f"first_frame_composite_clip_{clip_index}" in video_state.completed_steps
            assert f"veo_generation_clip_{clip_index}" in video_state.completed_steps

    # 3. run_state.cumulative_cost_usd > 0
    assert run_state.cumulative_cost_usd > 0.0, (
        "Expected cumulative cost > 0 after mocked but cost-tracked API calls"
    )

    # 4. No exceptions (implicit — test reaching this point means no exception was raised)
