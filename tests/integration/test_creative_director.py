"""Integration test for the creative director step.

Requires Vertex AI credentials (GOOGLE_CLOUD_PROJECT + GOOGLE_APPLICATION_CREDENTIALS)
and UGC_RUN_PAID_TESTS=1 to run. Uses an inline ProductBrief fixture (no vision call needed).
"""

from __future__ import annotations

import os
import pathlib
import uuid
from datetime import datetime, timezone

import pytest

from ugc_pipeline.models import ProductBrief, RunState, VideoSpec
from ugc_pipeline.steps.product_analyst import make_default_client
from ugc_pipeline.steps.creative_director import run_creative_director


_TALENT_POOL: dict = {
    "talent_02": {
        "gender": "woman",
        "age_range": "late 20s",
        "aesthetic": "natural, relaxed, warm",
        "camera_relationship": "conversational",
        "lighting_preference": "soft natural window light",
    }
}

_FIXTURE_BRIEF = ProductBrief(
    product_id="a3f9c12e7b04",
    image_path="assets/products/a3f9c12e7b04.jpg",
    shape="cylindrical mug with a C-shaped handle",
    dominant_colours=["#F5F0E8", "#3B2A1A", "#FFFFFF"],
    packaging_style="kraft paper box with embossed geometric pattern and ribbon closure",
    inferred_category="kitchenware / home lifestyle",
    lifestyle_contexts=[
        "morning routine at a kitchen counter",
        "desk setup during remote work",
        "outdoor picnic on a blanket",
    ],
    visual_notes="Matte ceramic surface with slight speckle texture.",
    created_at=datetime(2026, 5, 8, 9, 0, 0, tzinfo=timezone.utc),
)


@pytest.mark.paid
async def test_creative_director_live(tmp_path: pathlib.Path) -> None:
    """Call the real Gemini 2.5 Pro text endpoint and validate the three VideoSpecs."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    creds_path = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS",
        os.environ.get("GOOGLE_DRIVE_CREDENTIALS_PATH", ""),
    )
    assert project, "GOOGLE_CLOUD_PROJECT must be set for paid integration tests"
    assert creds_path, "GOOGLE_APPLICATION_CREDENTIALS must point at the service account JSON"

    client = make_default_client(project, location, creds_path)
    run_state = RunState(
        run_id="integration-cd-run",
        started_at=datetime.now(timezone.utc),
    )

    specs = await run_creative_director(
        _FIXTURE_BRIEF,
        _TALENT_POOL,
        client=client,
        run_state=run_state,
        state_root=tmp_path / "state",
        clip_counts=(2, 3, 2),
    )

    assert len(specs) == 3

    tones = [s.tone for s in specs]
    assert len(set(tones)) == 3, "All three specs must have distinct tones"

    for i, spec in enumerate(specs):
        assert isinstance(spec, VideoSpec)
        assert spec.product_id == _FIXTURE_BRIEF.product_id
        assert spec.spec_index == i
        # script_blocks should be non-empty Italian-ish text
        for block in spec.script_blocks:
            assert len(block) > 10, f"script_block too short: {block!r}"
        assert len(spec.script_blocks) == spec.clip_count
        assert len(spec.scene_descriptions) == spec.clip_count

    assert run_state.cumulative_cost_usd > 0
