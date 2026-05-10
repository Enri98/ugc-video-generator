"""Creative director step for the UGC pipeline.

Calls Gemini 2.5 Pro (text) three times — once per VideoSpec — and returns
a list of three VideoSpec objects for the given product. See SPEC.md §5
step 3 for the full contract.

Pricing constants (estimates, per SPEC.md §12):
  Gemini 2.5 Pro text input:  $1.25 / 1 000 000 tokens
  Gemini 2.5 Pro text output: $10.00 / 1 000 000 tokens
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Any

import pydantic
import structlog
import tenacity

from ugc_pipeline.cost_tracker import increment_cost
from ugc_pipeline.models import ProductBrief, PromptVersion, RunState, VideoSpec, VideoState
from ugc_pipeline.prompts import director as director_prompt
from ugc_pipeline.prompts.director import TONES
from ugc_pipeline.state_manager import (
    _video_spec_path,
    load_video_spec,
    save_video_spec,
    save_video_state,
)
from ugc_pipeline.utils.config import get_talent_descriptor

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pricing (estimates per SPEC.md §12)
# ---------------------------------------------------------------------------

_INPUT_COST_PER_TOKEN_USD: float = 1.25 / 1_000_000
_OUTPUT_COST_PER_TOKEN_USD: float = 10.00 / 1_000_000
_FLAT_ESTIMATE_USD: float = 0.01  # used before response is available


def estimate_creative_director_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost for a creative director call using Gemini 2.5 Pro text rates.

    These are rough estimates per SPEC.md §12.
    Input rate:  $1.25 / M tokens.
    Output rate: $10.00 / M tokens.
    """
    return (
        input_tokens * _INPUT_COST_PER_TOKEN_USD
        + output_tokens * _OUTPUT_COST_PER_TOKEN_USD
    )


# ---------------------------------------------------------------------------
# Idempotency scan helper
# ---------------------------------------------------------------------------


def _find_existing_spec(
    product_id: str,
    spec_index: int,
    state_root: pathlib.Path,
) -> VideoSpec | None:
    """Scan state/videos/ for a spec matching product_id and spec_index.

    Returns the loaded VideoSpec if found, otherwise None.
    """
    videos_dir = state_root / "videos"
    if not videos_dir.is_dir():
        return None
    for spec_file in videos_dir.glob("*.spec.json"):
        try:
            data = json.loads(spec_file.read_text(encoding="utf-8"))
            if data.get("product_id") == product_id and data.get("spec_index") == spec_index:
                return VideoSpec.model_validate(data)
        except Exception:  # noqa: BLE001
            continue
    return None


# ---------------------------------------------------------------------------
# Tenacity-wrapped API call
# ---------------------------------------------------------------------------


def _make_api_caller(client: Any, model: str, contents: list[Any], config: Any) -> Any:
    """Return an awaitable for the Gemini text API call, wrapped in tenacity retries."""

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
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
# Corrective-retry prompt builder
# ---------------------------------------------------------------------------


def _render_corrective_prompt(
    original_prompt: str,
    validation_error_msg: str,
    rejected_script_blocks: list[str],
    speaking_clip_index: int,
) -> str:
    """Build a corrective follow-up prompt when the first response fails VideoSpec validation.

    Parameters
    ----------
    original_prompt:
        The full rendered director prompt that was sent in the first call.
    validation_error_msg:
        The string representation of the pydantic ValidationError.
    rejected_script_blocks:
        The script_blocks list extracted from the rejected response, or [] if JSON
        parsing failed entirely.
    speaking_clip_index:
        The 0-based index of the single clip that is allowed to have spoken dialogue.
    """
    if rejected_script_blocks:
        blocks_repr = "\n".join(
            f"  [{i}]: {repr(block)}" for i, block in enumerate(rejected_script_blocks)
        )
        blocks_section = (
            f"Your previous (rejected) script_blocks were:\n{blocks_repr}\n"
        )
    else:
        blocks_section = (
            "Your previous response could not be parsed as JSON, so the individual "
            "script_blocks entries could not be extracted.\n"
        )

    correction_section = (
        "\n\n--- CORRECTION REQUIRED ---\n"
        "Your previous response was REJECTED by the schema validator.\n"
        f"The error was: {validation_error_msg}\n\n"
        "The most common mistake is putting spoken text in MORE THAN ONE script_blocks "
        f"entry — only script_blocks[{speaking_clip_index}] may contain Italian voice; "
        "all others MUST be the literal string '[silent]'.\n"
        f"{blocks_section}"
        "Re-emit a corrected VideoSpec JSON now."
    )

    return original_prompt + correction_section


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_creative_director(
    brief: ProductBrief,
    talent_pool: dict[str, dict[str, Any]],
    *,
    client: Any,
    run_state: RunState,
    state_root: pathlib.Path,
    global_max_usd: float = 50.0,
    clip_counts: tuple[int, ...] = (2, 3, 2),
    brand_guidance: dict | None = None,
    speaking_clip_index: int = 1,
) -> list[VideoSpec]:
    """Run the creative director step and return three VideoSpec objects.

    Idempotent per spec: if a spec for product_id + spec_index already exists on
    disk it is reused without an API call.

    Parameters
    ----------
    brief:
        The ProductBrief for the product being directed.
    talent_pool:
        Mapping of talent_id -> descriptor dict from talent_pool.yaml.
    client:
        An object satisfying GeminiClientProtocol (real or mock).
    run_state:
        The current RunState; cumulative cost is incremented per call.
    state_root:
        Root of the state directory tree.
    global_max_usd:
        Kill-switch threshold forwarded to increment_cost.
    clip_counts:
        Number of clips for each of the three specs respectively.
    speaking_clip_index:
        0-based index of the single clip that carries spoken Italian voiceover.
        Passed through to director_prompt.render() and used for corrective retries.
    """
    import time

    t_start_total = time.monotonic()
    sorted_talent_ids = sorted(talent_pool.keys())
    specs: list[VideoSpec] = []

    for spec_index in range(len(clip_counts)):
        # Idempotency: check whether a spec already exists for this position
        existing = _find_existing_spec(brief.product_id, spec_index, state_root)
        if existing is not None:
            log.info(
                "step_skipped_idempotent",
                step="creative_director",
                product_id=brief.product_id,
                spec_index=spec_index,
                video_id=existing.video_id,
            )
            specs.append(existing)
            continue

        t_start = time.monotonic()

        # Choose tone and talent deterministically
        tone = TONES[spec_index]
        other_tones = [t for t in TONES if t != tone]
        talent_id = sorted_talent_ids[spec_index % len(sorted_talent_ids)]
        talent_descriptor = get_talent_descriptor(talent_id, talent_pool)
        lifestyle_context = brief.lifestyle_contexts[spec_index % len(brief.lifestyle_contexts)]
        clip_count = clip_counts[spec_index]

        # Render prompt
        rendered_prompt = director_prompt.render(
            brief=brief,
            talent_id=talent_id,
            talent_descriptor=talent_descriptor,
            tone=tone,
            other_tones=other_tones,
            clip_count=clip_count,
            lifestyle_context=lifestyle_context,
            brand_guidance=brand_guidance,
            speaking_clip_index=speaking_clip_index,
            product_name=brief.product_name,
        )
        prompt_version = PromptVersion(
            step_name="creative_director",
            version=director_prompt.VERSION,
            content_sha256=hashlib.sha256(rendered_prompt.encode()).hexdigest(),
            rendered_at=datetime.now(timezone.utc),
        )

        # Build request
        contents = [{"role": "user", "parts": [{"text": rendered_prompt}]}]
        try:
            from google.genai.types import GenerateContentConfig  # type: ignore[import-untyped]
            config = GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VideoSpec,
            )
        except ImportError:
            config = None  # type: ignore[assignment]

        log.info(
            "step_started",
            step="creative_director",
            product_id=brief.product_id,
            spec_index=spec_index,
            tone=tone,
        )

        # 4. Increment cost estimate BEFORE awaiting (per SPEC.md §12 protocol)
        increment_cost(run_state, None, "creative_director_usd", _FLAT_ESTIMATE_USD, global_max_usd)

        response = await _make_api_caller(client, "gemini-2.5-pro", contents, config)

        # Refine cost using actual token counts if available
        refined_cost: float | None = None
        try:
            usage = response.usage_metadata  # type: ignore[union-attr]
            in_tok = getattr(usage, "prompt_token_count", 0) or 0
            out_tok = getattr(usage, "candidates_token_count", 0) or 0
            if in_tok or out_tok:
                refined_cost = estimate_creative_director_cost_usd(in_tok, out_tok)
                delta = refined_cost - _FLAT_ESTIMATE_USD
                if delta != 0.0:
                    increment_cost(run_state, None, "creative_director_usd", delta, global_max_usd)
        except Exception:  # noqa: BLE001
            pass

        cost_this_call = refined_cost if refined_cost is not None else _FLAT_ESTIMATE_USD

        # Parse and validate response; attempt one corrective retry on ValidationError
        response_text: str = response.text  # type: ignore[union-attr]
        corrective_prompt_version: PromptVersion | None = None
        try:
            spec = VideoSpec.model_validate_json(response_text)
        except pydantic.ValidationError as val_exc:
            # Extract rejected script_blocks if JSON is parseable
            try:
                rejected_script_blocks: list[str] = json.loads(response_text).get("script_blocks", [])
            except (json.JSONDecodeError, Exception):
                rejected_script_blocks = []

            log.warning(
                "creative_director_validation_failed",
                product_id=brief.product_id,
                spec_index=spec_index,
                error=str(val_exc),
                response_excerpt=response_text[:500],
            )

            # Build corrective prompt and submit ONE retry
            corrective_prompt = _render_corrective_prompt(
                original_prompt=rendered_prompt,
                validation_error_msg=str(val_exc),
                rejected_script_blocks=rejected_script_blocks,
                speaking_clip_index=speaking_clip_index,
            )
            corrective_prompt_version = PromptVersion(
                step_name="creative_director_corrective",
                version=director_prompt.VERSION,
                content_sha256=hashlib.sha256(corrective_prompt.encode()).hexdigest(),
                rendered_at=datetime.now(timezone.utc),
            )
            corrective_contents = [{"role": "user", "parts": [{"text": corrective_prompt}]}]

            # Bill BEFORE awaiting the corrective call (same "bill on submit" rule as above)
            increment_cost(run_state, None, "creative_director_usd", _FLAT_ESTIMATE_USD, global_max_usd)

            corrective_response = await _make_api_caller(
                client, "gemini-2.5-pro", corrective_contents, config
            )

            # Refine cost using actual token counts from the corrective response
            try:
                corrective_usage = corrective_response.usage_metadata  # type: ignore[union-attr]
                corr_in_tok = getattr(corrective_usage, "prompt_token_count", 0) or 0
                corr_out_tok = getattr(corrective_usage, "candidates_token_count", 0) or 0
                if corr_in_tok or corr_out_tok:
                    corr_refined = estimate_creative_director_cost_usd(corr_in_tok, corr_out_tok)
                    corr_delta = corr_refined - _FLAT_ESTIMATE_USD
                    if corr_delta != 0.0:
                        increment_cost(run_state, None, "creative_director_usd", corr_delta, global_max_usd)
            except Exception:  # noqa: BLE001
                pass

            corrective_response_text: str = corrective_response.text  # type: ignore[union-attr]
            # If this also raises ValidationError, propagate — no further retries
            spec = VideoSpec.model_validate_json(corrective_response_text)
        except Exception as exc:
            log.error(
                "step_failed",
                step="creative_director",
                product_id=brief.product_id,
                spec_index=spec_index,
                error=str(exc),
            )
            raise

        # Override fields that the orchestrator controls
        spec = spec.model_copy(
            update={
                "video_id": str(uuid.uuid4()),
                "product_id": brief.product_id,
                "spec_index": spec_index,
                "talent_id": talent_id,
                "created_at": datetime.now(timezone.utc),
                "speaking_clip_index": speaking_clip_index,
            }
        )

        # Validate clip-level list lengths
        if len(spec.scene_descriptions) != spec.clip_count:
            raise ValueError(
                f"spec_index={spec_index}: scene_descriptions length "
                f"({len(spec.scene_descriptions)}) does not match clip_count "
                f"({spec.clip_count})."
            )
        if len(spec.script_blocks) != spec.clip_count:
            raise ValueError(
                f"spec_index={spec_index}: script_blocks length "
                f"({len(spec.script_blocks)}) does not match clip_count "
                f"({spec.clip_count})."
            )

        # Persist spec
        save_video_spec(spec, root=state_root)

        # Build prompt_versions dict (include corrective entry if a retry was made)
        prompt_versions_dict: dict[str, PromptVersion] = {"creative_director": prompt_version}
        if corrective_prompt_version is not None:
            prompt_versions_dict["creative_director_corrective"] = corrective_prompt_version

        # Initialise a pending VideoState for this spec
        video_state = VideoState(
            video_id=spec.video_id,
            product_id=spec.product_id,
            spec_index=spec.spec_index,
            status="pending",
            prompt_versions=prompt_versions_dict,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        save_video_state(video_state, root=state_root)

        duration_ms = int((time.monotonic() - t_start) * 1000)
        log.info(
            "step_completed",
            step="creative_director",
            product_id=brief.product_id,
            spec_index=spec_index,
            video_id=spec.video_id,
            tone=tone,
            duration_ms=duration_ms,
            cost_usd_this_call=cost_this_call,
            cost_usd_cumulative=run_state.cumulative_cost_usd,
            prompt_versions={k: v.model_dump(mode="json") for k, v in prompt_versions_dict.items()},
        )

        specs.append(spec)

    return specs
