"""Unit tests for the product_analyst step (mocked Gemini client).

All tests are offline — no real API calls are made.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ugc_pipeline.models import ProductBrief, RunState
from ugc_pipeline.steps.product_analyst import (
    derive_product_name,
    estimate_product_analyst_cost_usd,
    run_product_analyst,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_state() -> RunState:
    return RunState(
        run_id=str(uuid.uuid4()),
        started_at=datetime.now(timezone.utc),
    )


def _valid_brief_dict(product_id: str = "aabbccddeeff") -> dict[str, Any]:
    return {
        "product_id": product_id,
        "image_path": f"assets/products/{product_id}.jpg",
        "shape": "cylindrical mug with a C-shaped handle",
        "dominant_colours": ["#F5F0E8", "#3B2A1A"],
        "packaging_style": "kraft paper box with embossed geometric pattern",
        "inferred_category": "kitchenware / home lifestyle",
        "lifestyle_contexts": [
            "morning routine at a kitchen counter",
            "desk setup during remote work",
            "outdoor picnic",
        ],
        "visual_notes": "matte ceramic surface with slight speckle texture",
        "created_at": "2026-05-08T09:00:00Z",
    }


def _make_mock_client(response_text: str) -> MagicMock:
    """Build a mock client whose generate_content returns a mock with .text."""
    usage_mock = MagicMock()
    usage_mock.prompt_token_count = 500
    usage_mock.candidates_token_count = 200

    response_mock = MagicMock()
    response_mock.text = response_text
    response_mock.usage_metadata = usage_mock

    client = MagicMock()
    client.generate_content = AsyncMock(return_value=response_mock)
    return client


# ---------------------------------------------------------------------------
# Test: pricing helper
# ---------------------------------------------------------------------------


def test_estimate_product_analyst_cost_sanity() -> None:
    cost = estimate_product_analyst_cost_usd(1_000_000, 1_000_000)
    # input: $1.25, output: $10.00 -> total $11.25
    assert abs(cost - 11.25) < 1e-6


def test_estimate_zero_tokens() -> None:
    assert estimate_product_analyst_cost_usd(0, 0) == 0.0


# ---------------------------------------------------------------------------
# Test 1: happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path(tmp_path: pathlib.Path) -> None:
    image_bytes = b"fake-image-data-for-testing"
    expected_product_id = hashlib.sha256(image_bytes).hexdigest()[:12]

    # Return a brief dict with a different product_id — the step must override it
    brief_dict = _valid_brief_dict("wrong_id_000")
    client = _make_mock_client(json.dumps(brief_dict))
    run_state = _make_run_state()

    state_root = tmp_path / "state"
    brief = await run_product_analyst(
        image_bytes,
        "product.jpg",
        None,
        client=client,
        run_state=run_state,
        state_root=state_root,
    )

    assert isinstance(brief, ProductBrief)
    assert brief.product_id == expected_product_id
    # State file must have been written
    state_file = state_root / "products" / f"{expected_product_id}.json"
    assert state_file.exists()
    # Image saved
    asset_file = tmp_path / "assets" / "products" / f"{expected_product_id}.jpg"
    assert asset_file.exists()
    # API was called exactly once
    client.generate_content.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 2: idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotency(tmp_path: pathlib.Path) -> None:
    image_bytes = b"idempotency-test-image-bytes"
    expected_product_id = hashlib.sha256(image_bytes).hexdigest()[:12]

    brief_dict = _valid_brief_dict()
    # Pre-populate with the correct product_id so load works
    brief_dict["product_id"] = expected_product_id
    brief_dict["image_path"] = f"assets/products/{expected_product_id}.jpg"
    client = _make_mock_client(json.dumps(brief_dict))
    run_state = _make_run_state()
    state_root = tmp_path / "state"

    # First call — hits the API
    brief1 = await run_product_analyst(
        image_bytes,
        "product.jpg",
        None,
        client=client,
        run_state=run_state,
        state_root=state_root,
    )
    assert client.generate_content.await_count == 1

    # Second call with same bytes — must NOT call the API again
    brief2 = await run_product_analyst(
        image_bytes,
        "product.jpg",
        None,
        client=client,
        run_state=run_state,
        state_root=state_root,
    )
    assert client.generate_content.await_count == 1, "API called more than once"
    assert brief2.product_id == brief1.product_id


# ---------------------------------------------------------------------------
# Test 3: cost is incremented in run_state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cost_incremented(tmp_path: pathlib.Path) -> None:
    image_bytes = b"cost-tracking-test-bytes"
    brief_dict = _valid_brief_dict()
    brief_dict["product_id"] = hashlib.sha256(image_bytes).hexdigest()[:12]
    brief_dict["image_path"] = "assets/products/dummy.jpg"
    client = _make_mock_client(json.dumps(brief_dict))
    run_state = _make_run_state()
    initial_cost = run_state.cumulative_cost_usd

    await run_product_analyst(
        image_bytes,
        "product.jpg",
        None,
        client=client,
        run_state=run_state,
        state_root=tmp_path / "state",
    )

    assert run_state.cumulative_cost_usd > initial_cost


# ---------------------------------------------------------------------------
# Test 4: malformed response raises after retries
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests for derive_product_name helper
# ---------------------------------------------------------------------------

_VARIANT_PATTERN = r"[-_ ]v\d+$"


def test_derive_product_name_strips_brand_and_variant_suffix() -> None:
    assert (
        derive_product_name(
            "acme-coffee-mug-v2.png", "Acme", variant_suffix_pattern=_VARIANT_PATTERN
        )
        == "Coffee Mug"
    )


def test_derive_product_name_case_insensitive_brand_match() -> None:
    assert (
        derive_product_name(
            "ACME-ceramic-cup-v1.jpg", "Acme", variant_suffix_pattern=_VARIANT_PATTERN
        )
        == "Ceramic Cup"
    )


def test_derive_product_name_no_brand_prefix() -> None:
    assert (
        derive_product_name(
            "kitchen-blend.png", "Acme", variant_suffix_pattern=_VARIANT_PATTERN
        )
        == "Kitchen Blend"
    )


def test_derive_product_name_brand_only_returns_empty() -> None:
    assert derive_product_name("Acme.png", "Acme") == ""


def test_derive_product_name_handles_underscores() -> None:
    assert (
        derive_product_name(
            "Acme_kitchen_blend_v1.jpg", "Acme", variant_suffix_pattern=_VARIANT_PATTERN
        )
        == "Kitchen Blend"
    )


def test_derive_product_name_no_brand_argument() -> None:
    assert derive_product_name("coffee-mug.png", "") == "Coffee Mug"


def test_derive_product_name_no_variant_suffix_pattern_keeps_full_stem() -> None:
    # Default None means no suffix stripping — the trailing token is preserved.
    assert derive_product_name("acme-coffee-mug-v2.png", "Acme") == "Coffee Mug V2"


# ---------------------------------------------------------------------------
# Test: run_product_analyst sets product_name from filename
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_product_analyst_sets_product_name_from_filename(tmp_path: pathlib.Path) -> None:
    image_bytes = b"product-name-derivation-test"
    expected_product_id = hashlib.sha256(image_bytes).hexdigest()[:12]

    brief_dict = _valid_brief_dict()
    brief_dict["product_id"] = expected_product_id
    brief_dict["image_path"] = f"assets/products/{expected_product_id}.jpg"
    client = _make_mock_client(json.dumps(brief_dict))
    run_state = _make_run_state()

    brief = await run_product_analyst(
        image_bytes,
        "acme-test-product-v1.png",
        None,
        client=client,
        run_state=run_state,
        state_root=tmp_path / "state",
        brand_name="Acme",
        variant_suffix_pattern=r"[-_ ]v\d+$",
    )

    assert brief.product_name == "Test Product"


@pytest.mark.asyncio
async def test_malformed_response_raises(tmp_path: pathlib.Path) -> None:
    image_bytes = b"bad-response-test"
    bad_response_mock = MagicMock()
    bad_response_mock.text = "this is not valid json {{{"
    bad_response_mock.usage_metadata = MagicMock(
        prompt_token_count=0, candidates_token_count=0
    )

    client = MagicMock()
    client.generate_content = AsyncMock(return_value=bad_response_mock)
    run_state = _make_run_state()

    # tenacity will retry 5 times and then reraise — the outer exception escapes
    with pytest.raises(Exception):
        await run_product_analyst(
            image_bytes,
            "product.jpg",
            None,
            client=client,
            run_state=run_state,
            state_root=tmp_path / "state",
        )
