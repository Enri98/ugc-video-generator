"""Unit tests for the creative_director step (mocked Gemini client).

All tests are offline — no real API calls are made.
"""

from __future__ import annotations

import json
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pydantic
import pytest

from ugc_pipeline.models import ProductBrief, RunState, VideoSpec
from ugc_pipeline.prompts.director import TONES
from ugc_pipeline.steps.creative_director import (
    _render_corrective_prompt,
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
        "script_blocks": ["" if i != 1 else f"Testo italiano per clip {i}." for i in range(clip_count)],
        "speaking_clip_index": 1,
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
        script_blocks=["", "Testo B."],
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
# Test 4: mismatched script_blocks length raises ValidationError after both attempts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mismatched_script_blocks_raises(tmp_path: pathlib.Path) -> None:
    brief = _make_brief()
    # clip_count=2 but only 1 script_block -> pydantic ValidationError
    # The corrective retry also returns the same bad data -> propagates ValidationError
    bad_spec = _make_spec_dict(0, clip_count=2)
    bad_spec["script_blocks"] = ["Solo un blocco."]  # length 1, clip_count 2

    client = _make_mock_client_multi([bad_spec, bad_spec])
    run_state = _make_run_state()

    with pytest.raises(pydantic.ValidationError):
        await run_creative_director(
            brief,
            _TALENT_POOL,
            client=client,
            run_state=run_state,
            state_root=tmp_path / "state",
            clip_counts=(2, 2, 2),
        )


# ---------------------------------------------------------------------------
# Test 5: ValidationError triggers corrective retry — succeeds on 2nd call
# ---------------------------------------------------------------------------


def _make_invalid_spec_dict(spec_index: int, clip_count: int = 2) -> dict[str, Any]:
    """Return a spec dict that violates the 'exactly one non-silent' rule."""
    return {
        "video_id": str(uuid.uuid4()),
        "product_id": "a3f9c12e7b04",
        "spec_index": spec_index,
        "tone": TONES[spec_index],
        "narrative_arc": f"Arc for spec {spec_index}.",
        "talent_id": "talent_01",
        "clip_count": clip_count,
        "scene_descriptions": [f"Scene {i}." for i in range(clip_count)],
        # All clips have spoken text — violates 'exactly one non-silent' rule
        "script_blocks": [f"Testo clip {i}." for i in range(clip_count)],
        "speaking_clip_index": 1,
        "visual_style_notes": "Natural light.",
        "created_at": "2026-05-08T09:00:00Z",
    }


@pytest.mark.asyncio
async def test_validation_error_triggers_corrective_retry(tmp_path: pathlib.Path) -> None:
    """First call returns invalid schema; second (corrective) call returns valid spec."""
    brief = _make_brief()
    run_state = _make_run_state()

    invalid_dict = _make_invalid_spec_dict(0, clip_count=2)
    valid_dict = _make_spec_dict(0, clip_count=2)

    # Need 3 total calls: 1 invalid + 1 corrective (for spec 0), then 2 more for specs 1 & 2
    spec_1_dict = _make_spec_dict(1, clip_count=2)
    spec_2_dict = _make_spec_dict(2, clip_count=2)

    responses = []
    for spec_dict in [invalid_dict, valid_dict, spec_1_dict, spec_2_dict]:
        usage_mock = MagicMock()
        usage_mock.prompt_token_count = 300
        usage_mock.candidates_token_count = 150
        resp = MagicMock()
        resp.text = json.dumps(spec_dict)
        resp.usage_metadata = usage_mock
        responses.append(resp)

    client = MagicMock()
    client.generate_content = AsyncMock(side_effect=responses)

    cost_before = run_state.cumulative_cost_usd

    specs = await run_creative_director(
        brief,
        _TALENT_POOL,
        client=client,
        run_state=run_state,
        state_root=tmp_path / "state",
        clip_counts=(2, 2, 2),
    )

    # (a) generate_content called 4 times total (1 invalid + 1 corrective + 2 for specs 1&2)
    assert client.generate_content.await_count == 4
    # (b) Corrective prompt (2nd call) must contain corrective markers
    second_call_args = client.generate_content.call_args_list[1]
    corrective_contents = second_call_args.kwargs.get("contents") or second_call_args.args[0] if second_call_args.args else second_call_args.kwargs["contents"]
    # contents is a list[dict]; extract text
    corrective_text = corrective_contents[0]["parts"][0]["text"]
    assert "REJECTED" in corrective_text
    assert "validator" in corrective_text
    # (c) Returned spec for index 0 is the valid one
    assert specs[0].product_id == brief.product_id
    assert len(specs) == 3
    # (d) Cost must have grown (corrective retry was billed)
    assert run_state.cumulative_cost_usd > cost_before


# ---------------------------------------------------------------------------
# Test 6: corrective retry also fails — ValidationError propagates, 2 calls total
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validation_error_corrective_retry_also_fails_propagates(tmp_path: pathlib.Path) -> None:
    """Both attempts return an invalid spec — ValidationError propagates after exactly 2 calls."""
    brief = _make_brief()
    run_state = _make_run_state()

    invalid_dict = _make_invalid_spec_dict(0, clip_count=2)

    responses = []
    for spec_dict in [invalid_dict, invalid_dict]:
        usage_mock = MagicMock()
        usage_mock.prompt_token_count = 300
        usage_mock.candidates_token_count = 150
        resp = MagicMock()
        resp.text = json.dumps(spec_dict)
        resp.usage_metadata = usage_mock
        responses.append(resp)

    client = MagicMock()
    client.generate_content = AsyncMock(side_effect=responses)

    with pytest.raises(pydantic.ValidationError):
        await run_creative_director(
            brief,
            _TALENT_POOL,
            client=client,
            run_state=run_state,
            state_root=tmp_path / "state",
            clip_counts=(2, 2, 2),
        )

    # Exactly 2 calls — no third attempt
    assert client.generate_content.await_count == 2


# ---------------------------------------------------------------------------
# Test 7: tenacity capped at 3 for transient errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tenacity_retries_capped_at_3_for_transient_errors(tmp_path: pathlib.Path) -> None:
    """First 2 calls raise ConnectionError; 3rd call succeeds. Total = 3 calls."""
    brief = _make_brief()
    run_state = _make_run_state()

    valid_dict = _make_spec_dict(0, clip_count=2)
    usage_mock = MagicMock()
    usage_mock.prompt_token_count = 300
    usage_mock.candidates_token_count = 150
    success_resp = MagicMock()
    success_resp.text = json.dumps(valid_dict)
    success_resp.usage_metadata = usage_mock

    call_count = {"n": 0}

    async def _side_effect(**kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise ConnectionError("transient network error")
        return success_resp

    client = MagicMock()
    client.generate_content = AsyncMock(side_effect=_side_effect)

    # Only run 1 spec to keep it simple
    specs = await run_creative_director(
        brief,
        _TALENT_POOL,
        client=client,
        run_state=run_state,
        state_root=tmp_path / "state",
        clip_counts=(2,),
    )

    assert len(specs) == 1
    assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# Test 8: speaking_clip_index propagated to director_prompt.render()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speaking_clip_index_propagated_to_render(tmp_path: pathlib.Path) -> None:
    """speaking_clip_index=2 is forwarded to director_prompt.render()."""
    brief = _make_brief()
    run_state = _make_run_state()

    # Build valid spec dicts for clip_count=3 with speaking_clip_index=2
    def _make_spec_3clip(idx: int) -> dict[str, Any]:
        return {
            "video_id": str(uuid.uuid4()),
            "product_id": "a3f9c12e7b04",
            "spec_index": idx,
            "tone": TONES[idx],
            "narrative_arc": f"Arc {idx}.",
            "talent_id": "talent_01",
            "clip_count": 3,
            "scene_descriptions": [f"Scene {i}." for i in range(3)],
            "script_blocks": ["[silent]", "[silent]", "Testo clip 2."],
            "speaking_clip_index": 2,
            "visual_style_notes": "Natural light.",
            "created_at": "2026-05-08T09:00:00Z",
        }

    spec_dicts = [_make_spec_3clip(i) for i in range(3)]
    client = _make_mock_client_multi(spec_dicts)

    captured_kwargs: list[dict] = []

    import ugc_pipeline.prompts.director as director_mod

    original_render = director_mod.render

    def _capturing_render(**kwargs: Any) -> str:
        captured_kwargs.append(kwargs)
        return original_render(**kwargs)

    with patch("ugc_pipeline.steps.creative_director.director_prompt.render", side_effect=_capturing_render):
        await run_creative_director(
            brief,
            _TALENT_POOL,
            client=client,
            run_state=run_state,
            state_root=tmp_path / "state",
            clip_counts=(3, 3, 3),
            speaking_clip_index=2,
        )

    assert len(captured_kwargs) == 3
    for call_kw in captured_kwargs:
        assert call_kw["speaking_clip_index"] == 2


# ---------------------------------------------------------------------------
# Test 9: corrective retry cost is billed to run_state (Bug #1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_corrective_retry_billed_to_run_state(tmp_path: pathlib.Path) -> None:
    """Corrective retry must increment cumulative_cost_usd a second time.

    Three measurement points:
      T0 — before any call
      T1 — captured at entry of the first generate_content call (flat billing already applied)
      T2 — captured at entry of the second (corrective) generate_content call
    Invariant: T0 < T1 < T2 (two distinct billing events, one per call).
    """
    brief = _make_brief()
    run_state = _make_run_state()

    invalid_dict = _make_invalid_spec_dict(0, clip_count=2)
    valid_dict = _make_spec_dict(0, clip_count=2)

    # Capture cost at the moment each generate_content invocation begins.
    # Billing occurs BEFORE the call (flat estimate + subsequent delta), so
    # the snapshot inside the mock reflects the post-flat-estimate state.
    cost_snapshots: list[float] = []

    def _make_resp(spec_dict: dict[str, Any]) -> MagicMock:
        usage = MagicMock()
        usage.prompt_token_count = 400
        usage.candidates_token_count = 200
        resp = MagicMock()
        resp.text = json.dumps(spec_dict)
        resp.usage_metadata = usage
        return resp

    call_count = {"n": 0}
    side_effects = [_make_resp(invalid_dict), _make_resp(valid_dict)]

    async def _side_effect(**kwargs: Any) -> Any:
        call_count["n"] += 1
        # Snapshot taken INSIDE generate_content — the flat billing for this call
        # has already been applied (it happens before the await).
        cost_snapshots.append(run_state.cumulative_cost_usd)
        return side_effects[call_count["n"] - 1]

    client = MagicMock()
    client.generate_content = AsyncMock(side_effect=_side_effect)

    t0 = run_state.cumulative_cost_usd

    # Only run spec_index=0 to keep the test focused
    await run_creative_director(
        brief,
        _TALENT_POOL,
        client=client,
        run_state=run_state,
        state_root=tmp_path / "state",
        clip_counts=(2,),
    )

    # generate_content was called exactly twice (initial + corrective)
    assert client.generate_content.await_count == 2

    # Two cost snapshots must have been captured (one per call)
    assert len(cost_snapshots) == 2, f"Expected 2 snapshots, got {len(cost_snapshots)}"

    # T0 < snapshot[0]: flat estimate was applied before first call
    assert cost_snapshots[0] > t0, (
        f"No billing occurred before first call: t0={t0}, snapshot[0]={cost_snapshots[0]}"
    )

    # snapshot[0] < snapshot[1]: flat estimate for corrective call was applied too
    assert cost_snapshots[1] > cost_snapshots[0], (
        "No billing occurred before the corrective retry call. "
        f"snapshot[0]={cost_snapshots[0]}, snapshot[1]={cost_snapshots[1]}"
    )

    # Overall cost must be positive and > T0
    assert run_state.cumulative_cost_usd > t0


# ---------------------------------------------------------------------------
# Test 10: speaking_clip_index overridden on returned spec (first attempt)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speaking_clip_index_overridden_on_returned_spec(tmp_path: pathlib.Path) -> None:
    """LLM returns speaking_clip_index=0; config passes speaking_clip_index=1.
    The returned spec must have speaking_clip_index=1."""
    brief = _make_brief()
    run_state = _make_run_state()

    # Spec dict where LLM claims speaking_clip_index=0
    llm_spec = {
        "video_id": str(uuid.uuid4()),
        "product_id": "a3f9c12e7b04",
        "spec_index": 0,
        "tone": TONES[0],
        "narrative_arc": "Arc 0.",
        "talent_id": "talent_01",
        "clip_count": 2,
        "scene_descriptions": ["Scene 0.", "Scene 1."],
        # Non-silent is at index 0 (LLM chose 0)
        "script_blocks": ["Testo italiano clip 0.", "[silent]"],
        "speaking_clip_index": 0,
        "visual_style_notes": "Natural light.",
        "created_at": "2026-05-08T09:00:00Z",
    }

    client = _make_mock_client_multi([llm_spec])

    specs = await run_creative_director(
        brief,
        _TALENT_POOL,
        client=client,
        run_state=run_state,
        state_root=tmp_path / "state",
        clip_counts=(2,),
        speaking_clip_index=1,  # config says 1
    )

    assert len(specs) == 1
    # Config must win — regardless of what LLM returned
    assert specs[0].speaking_clip_index == 1, (
        f"Expected speaking_clip_index=1 (from config) but got {specs[0].speaking_clip_index}"
    )


# ---------------------------------------------------------------------------
# Test 11: speaking_clip_index overridden on corrective retry spec
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speaking_clip_index_overridden_on_corrective_retry_spec(tmp_path: pathlib.Path) -> None:
    """First attempt fails validation; corrective succeeds with speaking_clip_index=2.
    Config passes speaking_clip_index=1. Returned spec must have speaking_clip_index=1."""
    brief = _make_brief()
    run_state = _make_run_state()

    # First attempt: all clips have spoken text → ValidationError
    invalid_dict = _make_invalid_spec_dict(0, clip_count=2)

    # Corrective response: LLM returns speaking_clip_index=2 (wrong)
    corrective_spec = {
        "video_id": str(uuid.uuid4()),
        "product_id": "a3f9c12e7b04",
        "spec_index": 0,
        "tone": TONES[0],
        "narrative_arc": "Arc 0.",
        "talent_id": "talent_01",
        "clip_count": 2,
        "scene_descriptions": ["Scene 0.", "Scene 1."],
        # Non-silent at index 1 but LLM claims speaking_clip_index=0
        "script_blocks": ["Testo italiano clip 0.", "[silent]"],
        "speaking_clip_index": 0,
        "visual_style_notes": "Natural light.",
        "created_at": "2026-05-08T09:00:00Z",
    }

    responses = []
    for spec_dict in [invalid_dict, corrective_spec]:
        usage = MagicMock()
        usage.prompt_token_count = 300
        usage.candidates_token_count = 150
        resp = MagicMock()
        resp.text = json.dumps(spec_dict)
        resp.usage_metadata = usage
        responses.append(resp)

    client = MagicMock()
    client.generate_content = AsyncMock(side_effect=responses)

    specs = await run_creative_director(
        brief,
        _TALENT_POOL,
        client=client,
        run_state=run_state,
        state_root=tmp_path / "state",
        clip_counts=(2,),
        speaking_clip_index=1,  # config says 1
    )

    assert len(specs) == 1
    assert client.generate_content.await_count == 2  # initial + corrective
    assert specs[0].speaking_clip_index == 1, (
        f"Expected speaking_clip_index=1 (from config) but got {specs[0].speaking_clip_index}"
    )
