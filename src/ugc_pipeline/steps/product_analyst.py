"""Product analyst step for the UGC pipeline.

Sends a product image to Gemini 2.5 Pro (vision) and returns a structured
ProductBrief. See SPEC.md §5 step 2 for the full contract.

Pricing constants (estimates, per SPEC.md §12 — tune empirically):
  Gemini 2.5 Pro vision input:  $1.25 / 1 000 000 tokens
  Gemini 2.5 Pro vision output: $10.00 / 1 000 000 tokens
"""

from __future__ import annotations

import base64
import hashlib
import json
import pathlib
from datetime import datetime, timezone
from typing import Any, Protocol

import structlog
import tenacity

from ugc_pipeline.cost_tracker import increment_cost
from ugc_pipeline.models import ProductBrief, PromptVersion, RunState
from ugc_pipeline.prompts import analyst as analyst_prompt
from ugc_pipeline.state_manager import (
    _product_brief_path,
    load_product_brief,
    save_product_brief,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pricing (estimates per SPEC.md §12)
# ---------------------------------------------------------------------------

_INPUT_COST_PER_TOKEN_USD: float = 1.25 / 1_000_000
_OUTPUT_COST_PER_TOKEN_USD: float = 10.00 / 1_000_000
_FLAT_ESTIMATE_USD: float = 0.02  # used before response is available


def estimate_product_analyst_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost for a product analyst call using Gemini 2.5 Pro vision rates.

    These are rough estimates per SPEC.md §12. Actual billing may differ.
    Input rate:  $1.25 / M tokens.
    Output rate: $10.00 / M tokens.
    """
    return (
        input_tokens * _INPUT_COST_PER_TOKEN_USD
        + output_tokens * _OUTPUT_COST_PER_TOKEN_USD
    )


# ---------------------------------------------------------------------------
# Client protocol (allows test injection without importing google.genai)
# ---------------------------------------------------------------------------


class GeminiClientProtocol(Protocol):
    """Minimal async interface for the Gemini generate_content call."""

    async def generate_content(
        self,
        *,
        model: str,
        contents: list[Any],
        config: object,
    ) -> object: ...


class _AioModelsAdapter:
    """Thin wrapper that holds a strong reference to the parent ``genai.Client``.

    Returning ``client.aio.models`` directly causes the parent client (and its
    async httpx pool) to be garbage-collected, which closes the connection
    mid-request. Holding the parent here keeps it alive for the caller's lifetime.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._models = client.aio.models

    async def generate_content(self, **kwargs: Any) -> Any:
        return await self._models.generate_content(**kwargs)


def make_default_client(
    project: str,
    location: str,
    credentials_path: str | pathlib.Path,
) -> Any:
    """Return an async Gemini client adapter authenticated via service account.

    Usage::

        client = make_default_client(
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ["GOOGLE_CLOUD_LOCATION"],
            credentials_path=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        )
        response = await client.generate_content(model=..., contents=..., config=...)
    """
    from google import genai  # type: ignore[import-untyped]
    from google.oauth2 import service_account  # type: ignore[import-untyped]

    credentials = service_account.Credentials.from_service_account_file(
        str(credentials_path),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    raw_client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
        credentials=credentials,
    )
    return _AioModelsAdapter(raw_client)


# ---------------------------------------------------------------------------
# MIME type helper
# ---------------------------------------------------------------------------

_EXT_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _mime_from_filename(filename: str) -> str:
    suffix = pathlib.Path(filename).suffix.lower()
    return _EXT_TO_MIME.get(suffix, "image/jpeg")


# ---------------------------------------------------------------------------
# Tenacity-wrapped API call
# ---------------------------------------------------------------------------


def _make_api_caller(client: GeminiClientProtocol, model: str, contents: list[Any], config: Any) -> Any:
    """Return an awaitable for the Gemini API call, wrapped in tenacity retries.

    Retry policy: up to 5 attempts, exponential back-off (multiplier=2, max=60s),
    re-raise the original exception on exhaustion.
    """

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=2, max=60),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call() -> Any:
        return await client.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

    return _call()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_product_analyst(
    image_bytes: bytes,
    filename: str,
    drive_file_id: str | None,
    *,
    client: GeminiClientProtocol,
    run_state: RunState,
    state_root: pathlib.Path,
    global_max_usd: float = 50.0,
) -> ProductBrief:
    """Run the product analyst step and return a ProductBrief.

    Idempotent: if a brief for this image already exists on disk, it is returned
    immediately without an API call.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the product image.
    filename:
        Original filename (used to infer MIME type and file extension).
    drive_file_id:
        Drive file ID of the source image, or None for local-only runs.
    client:
        An object satisfying GeminiClientProtocol (real or mock).
    run_state:
        The current RunState; cumulative cost is incremented here.
    state_root:
        Root of the state directory tree (e.g. Path("state")).
    global_max_usd:
        Kill-switch threshold forwarded to increment_cost.
    """
    import time

    t_start = time.monotonic()

    # 1. Deterministic product_id from image content
    product_id = hashlib.sha256(image_bytes).hexdigest()[:12]

    log.info(
        "step_started",
        step="product_analyst",
        product_id=product_id,
        filename=filename,
        drive_file_id=drive_file_id,
    )

    # 2. Idempotency check
    brief_path = _product_brief_path(product_id, state_root)
    if brief_path.exists():
        brief = load_product_brief(product_id, state_root)
        log.info(
            "step_skipped_idempotent",
            step="product_analyst",
            product_id=product_id,
        )
        return brief

    # 3. Render prompt and build PromptVersion
    rendered_prompt = analyst_prompt.render()
    prompt_version = PromptVersion(
        step_name="product_analyst",
        version=analyst_prompt.VERSION,
        content_sha256=hashlib.sha256(rendered_prompt.encode()).hexdigest(),
        rendered_at=datetime.now(timezone.utc),
    )

    # 4. Save image locally
    ext = pathlib.Path(filename).suffix.lower() or ".jpg"
    assets_dir = state_root.parent / "assets" / "products"
    assets_dir.mkdir(parents=True, exist_ok=True)
    image_path = assets_dir / f"{product_id}{ext}"
    image_path.write_bytes(image_bytes)

    # 5. Build Gemini request contents
    mime_type = _mime_from_filename(filename)
    image_b64 = base64.standard_b64encode(image_bytes).decode()
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": rendered_prompt},
                {"inline_data": {"mime_type": mime_type, "data": image_b64}},
            ],
        }
    ]

    # Build config with response_schema
    try:
        from google.genai.types import GenerateContentConfig  # type: ignore[import-untyped]
        config = GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ProductBrief,
        )
    except ImportError:
        # Allow running without google-genai installed (tests inject a mock config)
        config = None  # type: ignore[assignment]

    # 6. Increment cost estimate BEFORE awaiting (per SPEC.md §12 protocol)
    increment_cost(run_state, None, "product_analyst_usd", _FLAT_ESTIMATE_USD, global_max_usd)

    # Make the API call with tenacity retry on the inner helper
    response = await _make_api_caller(client, "gemini-2.5-pro", contents, config)

    # Refine cost estimate using actual token counts if available
    refined_cost: float | None = None
    try:
        usage = response.usage_metadata  # type: ignore[union-attr]
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        if in_tok or out_tok:
            refined_cost = estimate_product_analyst_cost_usd(in_tok, out_tok)
            # Adjust the already-recorded estimate by the delta
            delta = refined_cost - _FLAT_ESTIMATE_USD
            if delta != 0.0:
                increment_cost(run_state, None, "product_analyst_usd", delta, global_max_usd)
    except Exception:  # noqa: BLE001
        pass  # Token metadata not available; keep flat estimate

    cost_this_call = refined_cost if refined_cost is not None else _FLAT_ESTIMATE_USD

    # 7. Parse response
    try:
        response_text: str = response.text  # type: ignore[union-attr]
        brief = ProductBrief.model_validate_json(response_text)
    except (json.JSONDecodeError, Exception) as exc:
        log.error(
            "step_failed",
            step="product_analyst",
            product_id=product_id,
            error=str(exc),
        )
        raise

    # Override fields that the model must not invent
    brief = brief.model_copy(
        update={
            "product_id": product_id,
            "image_path": str(image_path),
            "created_at": datetime.now(timezone.utc),
        }
    )

    # 8. Persist brief
    save_product_brief(brief, root=state_root)

    duration_ms = int((time.monotonic() - t_start) * 1000)
    log.info(
        "step_completed",
        step="product_analyst",
        product_id=product_id,
        duration_ms=duration_ms,
        cost_usd_this_call=cost_this_call,
        cost_usd_cumulative=run_state.cumulative_cost_usd,
        prompt_versions={"product_analyst": prompt_version.model_dump(mode="json")},
    )

    return brief
