"""Integration test for the product analyst step.

Requires a live GOOGLE_API_KEY and UGC_RUN_PAID_TESTS=1 to run.
Uses a PIL-generated brand-neutral PNG so no real product image is needed.
"""

from __future__ import annotations

import hashlib
import os
import pathlib

import pytest

from ugc_pipeline.models import ProductBrief, RunState
from ugc_pipeline.steps.product_analyst import make_default_client, run_product_analyst


@pytest.mark.paid
async def test_product_analyst_live(
    sample_product_image_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Call the real Gemini 2.5 Pro vision endpoint and validate the returned brief."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    assert api_key, "GOOGLE_API_KEY must be set for paid integration tests"

    image_bytes = sample_product_image_path.read_bytes()
    expected_product_id = hashlib.sha256(image_bytes).hexdigest()[:12]

    client = make_default_client(api_key)
    run_state = RunState(
        run_id="integration-test-run",
        started_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )

    brief = await run_product_analyst(
        image_bytes,
        sample_product_image_path.name,
        None,
        client=client,
        run_state=run_state,
        state_root=tmp_path / "state",
    )

    assert isinstance(brief, ProductBrief)
    assert brief.product_id == expected_product_id
    assert brief.lifestyle_contexts, "lifestyle_contexts must be non-empty"
    assert len(brief.lifestyle_contexts) >= 1
    assert brief.shape
    assert brief.inferred_category
    assert run_state.cumulative_cost_usd > 0
