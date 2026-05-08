"""Unit tests for the creative_director step (mocked Gemini client).

All tests are offline — no real API calls are made.
"""

from __future__ import annotations

import json
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ugc_pipeline.models import ProductBrief, RunState, VideoSpec
from ugc_pipeline.prompts.director import TONES
from ugc_pipeline.steps.creative_director import (
    estimate_creative_director_cost_usd,
    run_creative_director,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TALENT_POOL: dict[str, dict[str, Any]] = {
    "talent_01": {
        "gender": "woman",
        "age_range": "early 30s",
        "aesthetic": "minimalist, calm",
        "camera_relationship": "direct address",
        "lighting_preference": "studio softbox",
    },
    "talent_02": {
        "gender": "woman",
        "age_range": "late 20s",
        "aesthetic": "natural, relaxed, warm",
        "camera_relationship": "conversational",
        "lighting_preference": "soft natural window light",
    },
}


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
            "outdoor picnic on a blanket",
        ],
        "visual_notes": "matte ceramic surface with slight speckle texture",
    }
    defaults.update(overrides)
    return ProductBrief(**defaults)  # type: ignore[arg-type]


def _make_run_state() -> RunState:
    return RunState(
        run_id=str(uuid.uuid4()),
        started_at=datetime.now(timezone.utc),
    )


def _make_spec_dict(spec_index: int, clip_count: int = 2, product_id: str = "a3f9c12e7b04") -> dict[str, Any]:
    return {
        "video_id": str(uuid.uuid4()),
        "product_id": product_id,
        "spec_index": spec_index,
        "tone": TONES[spec_index],
        "narrative_arc": f"A quiet narrative arc for spec {spec_index}.",
        "talent_id": "talent_01",
        "clip_count": clip_count,
        "scene_descriptions": [f"Scene {i} description." for i in range(clip_count)],
        "script_blocks": [f"Testo italiano per clip {i}." for i in range(clip_count)],
        "visual_style_notes": "Warm amber tone grading.",
        "created_at": "2026-05-08T09:00:00Z",
    }


def _make_mock_client_multi(spec_dicts: list[dict[str, Any]]) -> MagicMock:
    """Build a mock client that returns spec_dicts in sequence."""
    responses = []
    for spec_dict in spec_dicts:
        usage_mock = MagicMock()
        usage_mock.prompt_token_count = 300
        usage_mock.candidates_token_count = 150
        resp = MagicMock()
        resp.text = json.dumps(spec_dict)
        resp.usage_metadata = usage_mock
        responses.append(resp)

    client = MagicMock()
    client.generate_content = AsyncMock(side_effect=responses)
    return client


# ---------------------------------------------------------------------------
# Test: pricing helper
# ---------------------------------------------------------------------------


def test_estimate_creative_director_cost_sanity() -> None:
    cost = estimate_creative_director_cost_usd(1_000_000, 1_000_000)
    assert abs(cost - 11.25) < 1e-6


# ---------------------------------------------------------------------------
# Test 1: happy path — 3 specs, distinct tones, correct indexing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_returns_three_specs(tmp_path: pathlib.Path) -> None:
    brief = _make_brief()
    spec_dicts = [_make_spec_dict(i) for i in range(3)]
    client = _make_mock_client_multi(spec_dicts)
    run_state = _make_run_state()

    specs = await run_creative_director(
        brief,
        _TALENT_POOL,
        client=client,
        run_state=run_state,
        state_root=tmp_path / "state",
        clip_counts=(2, 2, 2),
    )

    assert len(specs) == 3
    for i, spec in enumerate(specs):
        assert isinstance(spec, VideoSpec)
        assert spec.product_id == brief.product_id
        assert spec.spec_index == i

    # Tones must be distinct
    tones = [s.tone for s in specs]
    assert len(set(tones)) == 3

    # video_ids must be distinct (orchestrator overrides them)
    video_ids = [s.video_id for s in specs]
    assert len(set(video_ids)) == 3

    # State files must exist
    state_dir = tmp_path / "state" / "videos"
    spec_files = list(state_dir.glob("*.spec.json"))
    assert len(spec_files) == 3


# ---------------------------------------------------------------------------
# Test 2: idempotency — pre-existing spec for spec_index=0 is reused
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotency_skips_existing_spec(tmp_path: pathlib.Path) -> None:
    brief = _make_brief()
    state_root = tmp_path / "state"

    # Pre-write a spec for spec_index=0
    pre_video_id = str(uuid.uuid4())
    pre_spec = VideoSpec(
        video_id=pre_video_id,
        product_id=brief.product_id,
        spec_index=0,
        tone=TONES[0],
        narrative_arc="Pre-existing arc.",
        talent_id="talent_01",
        clip_count=2,
        scene_descriptions=["Scene A.", "Scene B."],
        script_blocks=["Testo A.", "Testo B."],
    )
    from ugc_pipeline.state_manager import save_video_spec
    save_video_spec(pre_spec, root=state_root)

    # Client only needs to serve 2 responses (for spec_index 1 and 2)
    spec_dicts = [_make_spec_dict(i) for i in range(1, 3)]
    client = _make_mock_client_multi(spec_dicts)
    run_state = _make_run_state()

    specs = await run_creative_director(
        brief,
        _TALENT_POOL,
        client=client,
        run_state=run_state,
        state_root=state_root,
        clip_counts=(2, 2, 2),
    )

    assert len(specs) == 3
    # Client called only twice (not three times)
    assert client.generate_content.await_count == 2
    # First spec must be the pre-existing one
    assert specs[0].video_id == pre_video_id


# ---------------------------------------------------------------------------
# Test 3: cost incremented for each API call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cost_incremented_thrice(tmp_path: pathlib.Path) -> None:
    brief = _make_brief()
    spec_dicts = [_make_spec_dict(i) for i in range(3)]
    client = _make_mock_client_multi(spec_dicts)
    run_state = _make_run_state()
    initial_cost = run_state.cumulative_cost_usd

    await run_creative_director(
        brief,
        _TALENT_POOL,
        client=client,
        run_state=run_state,
        state_root=tmp_path / "state",
        clip_counts=(2, 2, 2),
    )

    # Cost must have increased three times (once per API call).
    # The mock has 300 input + 150 output tokens; refined cost ≈ 0.001875 per call.
    from ugc_pipeline.steps.creative_director import estimate_creative_director_cost_usd
    expected_min = initial_cost + 3 * estimate_creative_director_cost_usd(300, 150)
    assert run_state.cumulative_cost_usd >= expected_min * 0.9  # 10% tolerance


# ---------------------------------------------------------------------------
# Test 4: mismatched script_blocks length raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mismatched_script_blocks_raises(tmp_path: pathlib.Path) -> None:
    brief = _make_brief()
    # clip_count=2 but only 1 script_block -> mismatch
    bad_spec = _make_spec_dict(0, clip_count=2)
    bad_spec["script_blocks"] = ["Solo un blocco."]  # length 1, clip_count 2

    client = _make_mock_client_multi([bad_spec])
    run_state = _make_run_state()

    with pytest.raises(ValueError, match="script_blocks"):
        await run_creative_director(
            brief,
            _TALENT_POOL,
            client=client,
            run_state=run_state,
            state_root=tmp_path / "state",
            clip_counts=(2, 2, 2),
        )
