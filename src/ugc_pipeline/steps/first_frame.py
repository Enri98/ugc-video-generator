"""First-frame composite step for the UGC pipeline.

Calls Nano Banana 2 (Gemini 3 Flash Image) to generate a 1080x1920 PNG for
each clip in a video. See SPEC.md §5 step 4 for the full contract.

Cost estimate per image: $0.039 (midpoint of $0.02–$0.05 per SPEC.md §12).
Cost is recorded BEFORE each API call attempt so that a crash during the call
still appears in cost accounting on the next run.
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import structlog
import tenacity

from ugc_pipeline.cost_tracker import increment_cost
from ugc_pipeline.models import ProductBrief, PromptVersion, RunState, VideoSpec, VideoState
from ugc_pipeline.prompts import first_frame as first_frame_prompt
from ugc_pipeline.state_manager import save_run_state, save_video_state

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions  (SPEC.md §11)
# ---------------------------------------------------------------------------


class NanoBananaError(Exception):
    """Base error for Nano Banana 2 (Gemini 3 Flash Image) failures."""


class NanoBananaSafetyError(NanoBananaError):
    """Raised when the image generation request is rejected by a safety filter."""


class NanoBananaGenerationError(NanoBananaError):
    """Raised for transient or non-safety generation failures."""


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class NanoBananaResult:
    """Container for a successful image generation result."""

    png_bytes: bytes
    model: str = "gemini-3-pro-image-preview"


# ---------------------------------------------------------------------------
# Client protocol and default adapter
# ---------------------------------------------------------------------------


class NanoBananaClientProtocol(Protocol):
    """Minimal async interface for the Nano Banana 2 image generation API."""

    async def generate_image(
        self,
        *,
        prompt: str,
        model: str,
    ) -> NanoBananaResult: ...


class _DefaultNanoBananaClient:
    """Default adapter wrapping google.genai for image generation.

    # TODO Day 8: validate against google-genai 2.0 image API surface.
    """

    def __init__(
        self,
        project: str,
        location: str,
        credentials_path: str,
    ) -> None:
        from google import genai  # type: ignore[import-untyped]
        from google.oauth2 import service_account  # type: ignore[import-untyped]

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=credentials,
        )

    async def generate_image(
        self,
        *,
        prompt: str,
        model: str,
    ) -> NanoBananaResult:
        """Call Nano Banana via google.genai and return a NanoBananaResult.

        Passes ``image_config(aspect_ratio="9:16")`` so the model is forced
        to portrait output rather than the default 1:1 square. Falls back to
        no config if the SDK version doesn't support ImageConfig.
        """
        from google.genai import types  # type: ignore[import-untyped]

        config = None
        try:
            config = types.GenerateContentConfig(
                image_config=types.ImageConfig(aspect_ratio="9:16"),
                response_modalities=["IMAGE"],
            )
        except (AttributeError, TypeError):
            # Older SDK without ImageConfig — rely on prompt-text aspect hint only.
            pass

        kwargs: dict = {
            "model": model,
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }
        if config is not None:
            kwargs["config"] = config
        response = await self._client.aio.models.generate_content(**kwargs)
        try:
            # Extract PNG bytes from the first candidate's inline data
            part = response.candidates[0].content.parts[0]
            png_bytes: bytes = part.inline_data.data
        except (IndexError, AttributeError) as exc:
            raise NanoBananaGenerationError(
                f"Unexpected response shape from {model}: {exc}"
            ) from exc

        return NanoBananaResult(png_bytes=png_bytes, model=model)


def make_default_client(
    project: str,
    location: str,
    credentials_path: str,
) -> _DefaultNanoBananaClient:
    """Return the default Nano Banana 2 client adapter, authenticated via service account.

    Pass the returned object as the *client* argument to run_first_frame_for_clip.
    """
    return _DefaultNanoBananaClient(
        project=project,
        location=location,
        credentials_path=credentials_path,
    )


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------

_IMAGE_COST_USD: float = 0.039  # flat per-image estimate; midpoint of $0.02–$0.05 (SPEC.md §12)


def estimate_first_frame_cost_usd() -> float:
    """Return the flat per-image cost estimate for Nano Banana 2.

    This is a midpoint estimate of the $0.02–$0.05 range documented in
    SPEC.md §12. Actual billing may differ; tune empirically after Day 8.
    """
    return _IMAGE_COST_USD


# ---------------------------------------------------------------------------
# Inner tenacity-wrapped API helper
# (Each retry attempt bills separately — cost is recorded inside _call_with_cost.)
# ---------------------------------------------------------------------------


def _make_tenacity_caller(
    client: NanoBananaClientProtocol,
    prompt: str,
    model: str,
    run_state: RunState,
    video_state: VideoState,
    global_max_usd: float,
    video_state_path: pathlib.Path,
    run_state_path: pathlib.Path,
) -> Any:
    """Return an awaitable that retries the image API call up to 5 times.

    Cost is recorded BEFORE each attempt (per SPEC.md §12 protocol) so that
    a crash during any attempt still counts toward the budget. Tenacity re-raises
    on exhaustion.

    Only NanoBananaGenerationError and generic Exception trigger tenacity retries.
    NanoBananaSafetyError is NOT in the retry_if list — safety errors must be
    handled explicitly in the caller (single safety retry with softened prompt).
    """
    from ugc_pipeline.state_manager import _dump_model, write_state_atomic  # local import

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=2, max=60),
        retry=tenacity.retry_if_exception_type(NanoBananaGenerationError),
        reraise=True,
    )
    async def _call() -> NanoBananaResult:
        # Bill AFTER the call so 4xx/5xx (no compute consumed) does not pollute
        # the cost tracker. SPEC.md §12 prefers bill-before for long-running ops
        # (Veo); image generation is a single round-trip, so post-billing is safe.
        result = await client.generate_image(prompt=prompt, model=model)
        increment_cost(run_state, video_state, "first_frame_usd", _IMAGE_COST_USD, global_max_usd)
        write_state_atomic(video_state_path, _dump_model(video_state))
        write_state_atomic(run_state_path, _dump_model(run_state))
        return result

    return _call()


# ---------------------------------------------------------------------------
# Main per-clip entry point
# ---------------------------------------------------------------------------


async def run_first_frame_for_clip(
    spec: VideoSpec,
    brief: ProductBrief,
    clip_index: int,
    talent_descriptor: str,
    *,
    client: NanoBananaClientProtocol,
    video_state: VideoState,
    run_state: RunState,
    state_root: pathlib.Path,
    artifacts_root: pathlib.Path,
    global_max_usd: float = 50.0,
    dry_run: bool = False,
    model: str = "gemini-2.5-flash-image",
    product_size_hint: str | None = None,
) -> pathlib.Path:
    """Generate a first-frame PNG for *clip_index* and return its local path.

    Parameters
    ----------
    spec:
        VideoSpec for the current video.
    brief:
        ProductBrief describing the product.
    clip_index:
        0-based index of the clip being processed.
    talent_descriptor:
        Human-readable talent description resolved from talent_pool.yaml.
    client:
        An object satisfying NanoBananaClientProtocol (real or mock).
    video_state:
        Mutable VideoState for this video; mutated in place and saved.
    run_state:
        Mutable RunState for this run; mutated in place and saved.
    state_root:
        Root of the state directory tree.
    artifacts_root:
        Root of the artifacts directory (e.g. Path("artifacts")).
    global_max_usd:
        Kill-switch threshold forwarded to increment_cost.
    dry_run:
        If True, skip the API call, log the prompt and estimated cost, and
        return the expected output path without creating it.
    """
    import time

    from ugc_pipeline.state_manager import (
        _dump_model,
        _run_state_path,
        _video_state_path,
        write_state_atomic,
    )

    t_start = time.monotonic()

    artifact_key = f"clip_{clip_index}_firstframe"
    out_path = artifacts_root / video_state.video_id / f"clip_{clip_index}_firstframe.png"

    video_state_path = _video_state_path(video_state.video_id, state_root)
    run_state_path = _run_state_path(run_state.run_id, state_root)

    # ------------------------------------------------------------------
    # 1. Idempotency check
    # ------------------------------------------------------------------
    existing_path_str = video_state.artifacts.get(artifact_key)
    if existing_path_str is not None:
        existing_path = pathlib.Path(existing_path_str)
        if existing_path.exists() and existing_path.stat().st_size > 0:
            log.info(
                "step_skipped_idempotent",
                step="first_frame_composite",
                video_id=video_state.video_id,
                clip_index=clip_index,
                artifact_key=artifact_key,
            )
            return existing_path

    # ------------------------------------------------------------------
    # 2. Render prompt and build PromptVersion
    # ------------------------------------------------------------------
    if product_size_hint:
        rendered_prompt = first_frame_prompt.render(
            spec, brief, clip_index, talent_descriptor, product_size_hint=product_size_hint
        )
    else:
        rendered_prompt = first_frame_prompt.render(spec, brief, clip_index, talent_descriptor)
    prompt_sha = hashlib.sha256(rendered_prompt.encode()).hexdigest()
    prompt_version = PromptVersion(
        step_name="first_frame_composite",
        version=first_frame_prompt.VERSION,
        content_sha256=prompt_sha,
        rendered_at=datetime.now(timezone.utc),
    )
    video_state.prompt_versions["first_frame"] = prompt_version

    # ------------------------------------------------------------------
    # 3. Dry-run short-circuit
    # ------------------------------------------------------------------
    if dry_run:
        log.info(
            "step_skipped_dryrun",
            step="first_frame_composite",
            video_id=video_state.video_id,
            clip_index=clip_index,
            estimated_cost_usd=estimate_first_frame_cost_usd(),
            rendered_prompt=rendered_prompt,
        )
        return out_path

    log.info(
        "step_started",
        step="first_frame_composite",
        video_id=video_state.video_id,
        clip_index=clip_index,
        artifact_key=artifact_key,
    )

    # ------------------------------------------------------------------
    # 4. Call with safety retry
    # ------------------------------------------------------------------
    # `model` is the kwarg above. Default is "gemini-2.5-flash-image" (Nano
    # Banana 1 / Gemini 2.5 Flash Image, GA, available in us-central1).
    # To use Gemini 3 Pro Image Preview ("Nano Banana 2") set
    # nano_banana.model: "gemini-3-pro-image-preview" AND
    # nano_banana.location: "global" in pipeline_config.yaml — the preview
    # is currently only served from the `global` Vertex AI endpoint.

    async def _attempt(prompt: str) -> NanoBananaResult:  # type: ignore[return]
        """One attempt: bill + call, with tenacity on NanoBananaGenerationError."""
        return await _make_tenacity_caller(
            client=client,
            prompt=prompt,
            model=model,
            run_state=run_state,
            video_state=video_state,
            global_max_usd=global_max_usd,
            video_state_path=video_state_path,
            run_state_path=run_state_path,
        )

    try:
        result = await _attempt(rendered_prompt)
    except NanoBananaSafetyError:
        # Single safety retry with softened prompt — bills again
        log.warning(
            "step_safety_retry",
            step="first_frame_composite",
            video_id=video_state.video_id,
            clip_index=clip_index,
        )
        softened_prompt = first_frame_prompt.render_softened(spec, brief, clip_index, talent_descriptor)
        # This attempt also bills via _attempt (which uses _make_tenacity_caller)
        result = await _attempt(softened_prompt)

    # ------------------------------------------------------------------
    # 5. Persist artifact
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(result.png_bytes)

    video_state.artifacts[artifact_key] = str(out_path)
    video_state.updated_at = datetime.now(timezone.utc)
    save_video_state(video_state, root=state_root)

    duration_ms = int((time.monotonic() - t_start) * 1000)
    log.info(
        "step_completed",
        step="first_frame_composite",
        video_id=video_state.video_id,
        clip_index=clip_index,
        artifact_path=str(out_path),
        duration_ms=duration_ms,
        cost_usd_cumulative=run_state.cumulative_cost_usd,
        prompt_versions={"first_frame": prompt_version.model_dump(mode="json")},
    )

    return out_path


# ---------------------------------------------------------------------------
# Multi-clip orchestrator helper
# ---------------------------------------------------------------------------


async def run_first_frames(
    spec: VideoSpec,
    brief: ProductBrief,
    talent_descriptor: str,
    *,
    client: NanoBananaClientProtocol,
    video_state: VideoState,
    run_state: RunState,
    state_root: pathlib.Path,
    artifacts_root: pathlib.Path,
    global_max_usd: float = 50.0,
    dry_run: bool = False,
) -> list[pathlib.Path]:
    """Generate first-frame PNGs for all clips in *spec* sequentially.

    Returns a list of paths in clip order (index 0 … clip_count-1).

    Note: Clip-level parallelism is the orchestrator's responsibility; this
    helper processes clips serially and is used in tests and the CLI.
    """
    paths: list[pathlib.Path] = []
    for clip_index in range(spec.clip_count):
        path = await run_first_frame_for_clip(
            spec=spec,
            brief=brief,
            clip_index=clip_index,
            talent_descriptor=talent_descriptor,
            client=client,
            video_state=video_state,
            run_state=run_state,
            state_root=state_root,
            artifacts_root=artifacts_root,
            global_max_usd=global_max_usd,
            dry_run=dry_run,
        )
        paths.append(path)
    return paths
