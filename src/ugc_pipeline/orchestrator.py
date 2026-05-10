"""Orchestrator — coordinates per-product, per-video, per-clip execution.

See SPEC.md §7 (resume algorithm), §8 (concurrency model), §12 (kill-switch).
"""

from __future__ import annotations

import asyncio
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from ugc_pipeline.cost_tracker import (
    BudgetExceededError,
    KillSwitchFiredError,
    admit_clip_batch,
    check_kill_switch,
    increment_cost,
)
from ugc_pipeline.models import ProductBrief, RunState, VideoSpec, VideoState
from ugc_pipeline.state_manager import (
    _dump_model,
    _video_state_path,
    _video_spec_path,
    load_video_spec,
    load_video_state,
    save_run_state,
    save_video_state,
    sweep_locks,
    write_state_atomic,
)
from ugc_pipeline.steps.veo import VeoOperation, drain_inflight_veo

# Module-level step imports (not inside functions) so that test patches work correctly.
from ugc_pipeline.steps.product_analyst import run_product_analyst
from ugc_pipeline.steps.creative_director import run_creative_director

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Rate-limited / semaphore-bounded client adapters
# ---------------------------------------------------------------------------


class RateLimitedGeminiClient:
    """Wraps a Gemini models client and acquires the aiolimiter before each call."""

    def __init__(self, inner: Any, limiter: Any) -> None:
        self._inner = inner
        self._limiter = limiter

    async def generate_content(self, **kwargs: Any) -> Any:
        async with self._limiter:
            return await self._inner.generate_content(**kwargs)

    # Forward any other attribute access to the inner client.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class RateLimitedNanoBananaClient:
    """Wraps a Nano Banana client and acquires the aiolimiter before each call."""

    def __init__(self, inner: Any, limiter: Any) -> None:
        self._inner = inner
        self._limiter = limiter

    async def generate_image(self, **kwargs: Any) -> Any:
        async with self._limiter:
            return await self._inner.generate_image(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class SemaphoreBoundedVeoClient:
    """Wraps a Veo client and acquires a semaphore before each submit call."""

    def __init__(self, inner: Any, semaphore: asyncio.Semaphore) -> None:
        self._inner = inner
        self._semaphore = semaphore

    async def submit(self, **kwargs: Any) -> str:
        async with self._semaphore:
            return await self._inner.submit(**kwargs)

    async def poll(self, operation_id: str) -> dict:
        return await self._inner.poll(operation_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Orchestrator context
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorContext:
    """Bundles all shared dependencies for an orchestrator run."""

    # API clients (raw, un-rate-limited; adapters built from these)
    gemini_pro_client: Any
    nano_banana_client: Any
    veo_client: Any
    flash_client: Any
    drive_client: Any | None

    # State / artifact directories
    state_root: pathlib.Path
    artifacts_root: pathlib.Path

    # Drive output folders
    output_videos_folder_id: str | None
    output_reports_folder_id: str | None

    # Pipeline config (raw dict from load_pipeline_config)
    cfg: dict[str, Any]

    # Talent pool (raw dict from load_talent_pool)
    talent_pool: dict[str, dict[str, Any]]

    # Brand guidance (raw dict from load_brand_guidance; empty when absent)
    brand_guidance: dict[str, Any] = field(default_factory=dict)

    # Operating mode
    dry_run: bool = False

    # Concurrency controls (created externally so tests can inject them)
    gemini_limiter: Any = None       # aiolimiter.AsyncLimiter
    nano_banana_limiter: Any = None  # aiolimiter.AsyncLimiter
    veo_semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(4))

    # In-flight Veo operations (appended by _step_veo; drained on kill-switch)
    in_flight_veo: list[VeoOperation] = field(default_factory=list)

    # Kill-switch event (set when run_state.kill_switch_fired becomes True)
    kill_switch_event: asyncio.Event = field(default_factory=asyncio.Event)

    def effective_gemini_client(self) -> Any:
        """Return the Gemini client, optionally wrapped in a rate limiter."""
        if self.gemini_limiter is not None:
            return RateLimitedGeminiClient(self.gemini_pro_client, self.gemini_limiter)
        return self.gemini_pro_client

    def effective_nano_banana_client(self) -> Any:
        """Return the Nano Banana client, optionally wrapped in a rate limiter."""
        if self.nano_banana_limiter is not None:
            return RateLimitedNanoBananaClient(self.nano_banana_client, self.nano_banana_limiter)
        return self.nano_banana_client

    def effective_veo_client(self) -> Any:
        """Return the Veo client, bounded by the semaphore."""
        return SemaphoreBoundedVeoClient(self.veo_client, self.veo_semaphore)


# ---------------------------------------------------------------------------
# Artifact existence check (SPEC.md §7 idempotency rule)
# ---------------------------------------------------------------------------


def _artifact_exists_for_step(state: VideoState, step_name: str) -> bool:
    """Return True if the expected output artifact(s) for *step_name* exist on disk.

    Maps step names to the artifact key(s) that indicate completion.
    """
    key_map: dict[str, list[str]] = {
        "first_frame_composite": [],   # per-clip; handled separately
        "veo_generation": [],          # per-clip; handled separately
        "end_of_clip_trim": [],        # per-clip; handled separately
        "stitch": ["stitched"],
        "caption": ["final"],
        "report_gen": ["report"],
        "drive_upload": [],            # idempotency via drive_video_file_id / drive_report_file_id
    }
    keys = key_map.get(step_name, [])
    for key in keys:
        path_str = state.artifacts.get(key)
        if path_str is None:
            return False
        p = pathlib.Path(path_str)
        if not p.exists() or p.stat().st_size == 0:
            return False
    return bool(keys)  # True only if at least one key was checked


# ---------------------------------------------------------------------------
# Step wrappers — thin async functions calling the existing step modules
# ---------------------------------------------------------------------------


async def _step_first_frame(
    video_state: VideoState,
    spec: VideoSpec,
    brief: ProductBrief,
    run_state: RunState,
    ctx: OrchestratorContext,
) -> None:
    """Run first_frame_composite for all clips in the video.

    Clip 0 runs first (serial gatekeeper); clips 1..N run in parallel anchored
    on clip 0's PNG bytes as a visual reference. This keeps talent/lighting
    consistent across clips within a video.

    The original source product image (brief.image_path) is passed as a visual
    reference to EVERY clip so the model renders the actual product appearance
    (including labels and packaging) rather than a generic shape.

    Edge cases handled:
    - Clip 0 absent from admitted batch (completed in a prior run): load its
      bytes from disk for the chain without re-generating.
    - dry_run: file does not exist on disk; skip read_bytes() for all clips.
    - Empty admitted batch: return immediately.
    - Single admitted clip (only clip 0): run it alone, no chaining.
    - Source image missing or empty: degrade gracefully with a warning.
    """
    from ugc_pipeline.steps.first_frame import run_first_frame_for_clip
    from ugc_pipeline.utils.config import get_talent_descriptor

    cfg = ctx.cfg
    global_max = float(cfg.get("budget", {}).get("global_max_usd", 50.0))
    clip_indices = list(range(spec.clip_count))

    # Budget admission (per SPEC.md §8)
    clip_indices = admit_clip_batch(video_state, clip_indices, run_state, cfg.get("budget", cfg))

    if not clip_indices:
        return

    talent_descriptor = get_talent_descriptor(spec.talent_id, ctx.talent_pool)
    client = ctx.effective_nano_banana_client()
    nb_cfg = cfg.get("nano_banana", {})
    nb_model = str(nb_cfg.get("model", "gemini-2.5-flash-image"))
    nb_size_hint = nb_cfg.get("product_size_hint") or None  # None -> prompt default

    # ------------------------------------------------------------------
    # Read source product image bytes (one read per video — used as reference
    # for all clips so the model renders the actual product, not a generic shape).
    # ------------------------------------------------------------------
    from ugc_pipeline.steps.first_frame import _mime_from_path

    source_image_bytes: bytes | None = None
    source_image_path = pathlib.Path(brief.image_path)
    # Derive MIME type from the actual file extension so PNG sources are not
    # misidentified as JPEG (matches the defensive pattern from the analyst step).
    source_image_mime_type: str = _mime_from_path(source_image_path)
    if not ctx.dry_run:
        if source_image_path.exists() and source_image_path.stat().st_size > 0:
            source_image_bytes = source_image_path.read_bytes()
        else:
            log.warning(
                "first_frame_source_image_missing",
                video_id=video_state.video_id,
                path=str(source_image_path),
            )

    common_kwargs: dict = dict(
        spec=spec,
        brief=brief,
        talent_descriptor=talent_descriptor,
        client=client,
        video_state=video_state,
        run_state=run_state,
        state_root=ctx.state_root,
        artifacts_root=ctx.artifacts_root,
        global_max_usd=global_max,
        dry_run=ctx.dry_run,
        model=nb_model,
        product_size_hint=nb_size_hint,
        source_image_bytes=source_image_bytes,
        source_image_mime_type=source_image_mime_type,
    )

    # ------------------------------------------------------------------
    # Resolve clip 0's reference bytes (the visual anchor for clips 1..N)
    # ------------------------------------------------------------------
    clip0_bytes: bytes | None = None
    indices_to_run_in_parallel = list(clip_indices)

    if 0 in clip_indices:
        # Run clip 0 first as the serial gatekeeper
        clip0_path = await run_first_frame_for_clip(
            clip_index=0, reference_image_bytes=None, **common_kwargs
        )
        indices_to_run_in_parallel.remove(0)
        # In dry_run, clip0_path does not exist on disk — skip read_bytes()
        if not ctx.dry_run:
            clip0_bytes = clip0_path.read_bytes()
    elif any(i >= 1 for i in clip_indices):
        # Clip 0 was completed in a prior run; load its bytes from disk for the chain.
        clip0_path_str = video_state.artifacts.get("clip_0_firstframe")
        if clip0_path_str and not ctx.dry_run:
            clip0_path = pathlib.Path(clip0_path_str)
            if clip0_path.exists() and clip0_path.stat().st_size > 0:
                clip0_bytes = clip0_path.read_bytes()
            else:
                log.warning(
                    "first_frame_chain_degraded",
                    reason="clip_0_artifact_missing_or_empty",
                    clip_0_path=clip0_path_str,
                    video_id=video_state.video_id,
                )

    if not indices_to_run_in_parallel:
        return

    log.info(
        "first_frame_chain_active",
        clip_count=len(indices_to_run_in_parallel),
        reference_source="clip_0",
        reference_available=clip0_bytes is not None,
        source_image_available=source_image_bytes is not None,
        video_id=video_state.video_id,
    )

    tasks = [
        run_first_frame_for_clip(
            clip_index=i, reference_image_bytes=clip0_bytes, **common_kwargs
        )
        for i in indices_to_run_in_parallel
    ]
    await asyncio.gather(*tasks)


async def _step_veo(
    video_state: VideoState,
    spec: VideoSpec,
    brief: ProductBrief,
    run_state: RunState,
    ctx: OrchestratorContext,
) -> None:
    """Run veo_generation for all clips in the video (parallel, semaphore-bounded)."""
    from ugc_pipeline.steps.veo import VeoOperation, run_veo_for_clip

    cfg = ctx.cfg
    veo_cfg = cfg.get("veo", {})
    global_max = float(cfg.get("budget", {}).get("global_max_usd", 50.0))
    poll_interval = float(veo_cfg.get("poll_interval_seconds", 15.0))
    poll_timeout = float(veo_cfg.get("poll_timeout_seconds", 360.0))

    clip_indices = list(range(spec.clip_count))
    # Budget admission (per SPEC.md §8)
    clip_indices = admit_clip_batch(video_state, clip_indices, run_state, cfg.get("budget", cfg))

    veo_client = ctx.effective_veo_client()

    async def _run_one(i: int) -> None:
        op = VeoOperation(operation_id="", clip_index=i, video_id=video_state.video_id)
        # Track in-flight for drain-on-kill-switch
        ctx.in_flight_veo.append(op)
        try:
            await run_veo_for_clip(
                spec,
                brief,
                i,
                client=veo_client,
                flash_client=ctx.flash_client,
                video_state=video_state,
                run_state=run_state,
                state_root=ctx.state_root,
                artifacts_root=ctx.artifacts_root,
                poll_interval_seconds=poll_interval,
                poll_timeout_seconds=poll_timeout,
                global_max_usd=global_max,
                dry_run=ctx.dry_run,
            )
        finally:
            # Remove from in-flight list once done
            try:
                ctx.in_flight_veo.remove(op)
            except ValueError:
                pass

    tasks = [_run_one(i) for i in clip_indices]
    await asyncio.gather(*tasks)


def _dry_run_skip(step_name: str, video_state: VideoState) -> None:
    """Emit a step_skipped_dryrun log event for a local post-production step.

    Per SPEC.md §11, the local steps (trim/stitch/caption) would normally run
    against a placeholder video in dry-run; v1 simply skips them since first_frame
    and veo were skipped upstream and there is no real artifact to operate on.
    """
    from ugc_pipeline.utils.logging import get_logger

    get_logger(__name__).info(
        "step_skipped_dryrun",
        step=step_name,
        video_id=video_state.video_id,
    )


async def _step_trim(
    video_state: VideoState,
    spec: VideoSpec,
    brief: ProductBrief,
    run_state: RunState,
    ctx: OrchestratorContext,
) -> None:
    """Run end_of_clip_trim for all clips in the video."""
    if ctx.dry_run:
        _dry_run_skip("end_of_clip_trim", video_state)
        return

    from ugc_pipeline.steps.trim import run_trim_for_clip

    cfg = ctx.cfg
    trim_tail = float(cfg.get("post_production", {}).get("trim_tail_seconds", 0.5))

    tasks = [
        run_trim_for_clip(
            video_state,
            i,
            artifacts_root=ctx.artifacts_root,
            trim_tail_seconds=trim_tail,
        )
        for i in range(spec.clip_count)
    ]
    await asyncio.gather(*tasks)


async def _step_stitch(
    video_state: VideoState,
    spec: VideoSpec,
    brief: ProductBrief,
    run_state: RunState,
    ctx: OrchestratorContext,
) -> None:
    """Run stitch for the video."""
    if ctx.dry_run:
        _dry_run_skip("stitch", video_state)
        return

    from ugc_pipeline.steps.stitch import cleanup_after_step, run_stitch

    transition_seconds = float(
        ctx.cfg.get("post_production", {}).get("transition_seconds", 0.0)
    )
    await run_stitch(
        video_state,
        artifacts_root=ctx.artifacts_root,
        transition_seconds=transition_seconds,
    )
    cleanup_after_step(video_state, "stitch", ctx.cfg.get("cleanup", {}))


async def _step_caption(
    video_state: VideoState,
    spec: VideoSpec,
    brief: ProductBrief,
    run_state: RunState,
    ctx: OrchestratorContext,
) -> None:
    """Run caption for the video."""
    if ctx.dry_run:
        _dry_run_skip("caption", video_state)
        return

    from ugc_pipeline.steps.caption import run_caption

    cfg = ctx.cfg
    pp = cfg.get("post_production", {})

    await run_caption(
        video_state,
        spec,
        artifacts_root=ctx.artifacts_root,
        max_chars_per_line=int(pp.get("caption_max_chars_per_line", 30)),
        video_w=int(pp.get("video_width", 720)),
        video_h=int(pp.get("video_height", 1280)),
        font=str(pp.get("caption_font", "Arial")),
        font_size=int(pp.get("caption_font_size", 44)),
    )


async def _step_report(
    video_state: VideoState,
    spec: VideoSpec,
    brief: ProductBrief,
    run_state: RunState,
    ctx: OrchestratorContext,
) -> None:
    """Run report_gen for the video."""
    from ugc_pipeline.steps.report_gen import run_report_gen

    await run_report_gen(
        video_state,
        spec,
        run_state,
        artifacts_root=ctx.artifacts_root,
        dry_run=ctx.dry_run,
    )


async def _step_drive_upload(
    video_state: VideoState,
    spec: VideoSpec,
    brief: ProductBrief,
    run_state: RunState,
    ctx: OrchestratorContext,
) -> None:
    """Run drive_upload for the video."""
    drive_cfg = ctx.cfg.get("drive_upload", {}) or {}
    if not bool(drive_cfg.get("enabled", True)):
        log.info(
            "step_skipped_disabled",
            step="drive_upload",
            video_id=video_state.video_id,
            reason="drive_upload.enabled is false",
        )
        # Mark the video as completed-local so downstream tooling can distinguish.
        video_state.status = "completed_local"
        video_state.updated_at = datetime.now(timezone.utc)
        save_video_state(video_state, root=ctx.state_root)
        return

    if ctx.dry_run:
        _dry_run_skip("drive_upload", video_state)
        return

    from ugc_pipeline.steps.drive_upload import run_drive_upload

    await run_drive_upload(
        video_state,
        client=ctx.drive_client,
        output_videos_folder_id=ctx.output_videos_folder_id,
        output_reports_folder_id=ctx.output_reports_folder_id,
    )


# ---------------------------------------------------------------------------
# Step pipeline declaration (SPEC.md §7)
# ---------------------------------------------------------------------------

STEP_PIPELINE: list[tuple[str, Any]] = [
    ("first_frame_composite", _step_first_frame),
    ("veo_generation",        _step_veo),
    ("end_of_clip_trim",      _step_trim),
    ("stitch",                _step_stitch),
    ("caption",               _step_caption),
    ("report_gen",            _step_report),
    ("drive_upload",          _step_drive_upload),
]


# ---------------------------------------------------------------------------
# Admit-product helper (SPEC.md §12 kill-switch + §13 test contract)
# ---------------------------------------------------------------------------


def admit_product(run_state: RunState, product_id: str) -> None:
    """Raise KillSwitchFiredError if the kill-switch has fired.

    Called before starting any new product in the main loop. This satisfies
    the SPEC.md §13 test_kill_switch_refuses_new_work contract.
    """
    check_kill_switch(run_state)


# ---------------------------------------------------------------------------
# Per-video processing  (SPEC.md §7 resume algorithm)
# ---------------------------------------------------------------------------


async def process_video(
    video_id: str,
    brief: ProductBrief,
    ctx: OrchestratorContext,
    run_state: RunState,
) -> None:
    """Drive one video through all pipeline steps, resuming from the last completed step.

    Implements the SPEC.md §7 resume algorithm pseudocode verbatim.
    """
    cfg = ctx.cfg
    state_root = ctx.state_root
    global_max = float(cfg.get("budget", {}).get("global_max_usd", 50.0))
    per_video_max = float(cfg.get("budget", {}).get("per_video_max_usd", 8.0))

    state = load_video_state(video_id, state_root)
    spec = load_video_spec(video_id, state_root)

    log.info("video_started", video_id=video_id, product_id=state.product_id, spec_index=spec.spec_index)

    for step_name, step_fn in STEP_PIPELINE:
        # 1. Already completed in a previous run → skip
        if step_name in state.completed_steps:
            log.info("step_skipped_idempotent", step=step_name, video_id=video_id)
            continue

        # 2. Artifact already exists → mark completed without re-running
        if _artifact_exists_for_step(state, step_name):
            log.info("step_skipped_idempotent", step=step_name, video_id=video_id, reason="artifact_exists")
            state.completed_steps.append(step_name)
            state.updated_at = datetime.now(timezone.utc)
            save_video_state(state, root=state_root)
            continue

        # 3. Mark in-progress
        state.current_step = step_name
        state.status = "in_progress"
        state.updated_at = datetime.now(timezone.utc)
        save_video_state(state, root=state_root)

        try:
            await step_fn(state, spec, brief, run_state, ctx)
        except BudgetExceededError as exc:
            log.error("step_failed", step=step_name, video_id=video_id, error=str(exc), status="failed_budget")
            state.status = "failed_budget"
            state.last_error = str(exc)
            state.updated_at = datetime.now(timezone.utc)
            save_video_state(state, root=state_root)
            save_run_state(run_state, root=state_root)
            run_state.videos_failed += 1
            return
        except KillSwitchFiredError:
            log.critical("kill_switch_fired", step=step_name, video_id=video_id)
            ctx.kill_switch_event.set()
            # Do not change video status — leave as in_progress so it resumes next run
            state.updated_at = datetime.now(timezone.utc)
            save_video_state(state, root=state_root)
            save_run_state(run_state, root=state_root)
            raise
        except Exception as exc:
            log.error("step_failed", step=step_name, video_id=video_id, error=str(exc))
            state.status = "failed_error"
            state.last_error = str(exc)
            state.updated_at = datetime.now(timezone.utc)
            save_video_state(state, root=state_root)
            save_run_state(run_state, root=state_root)
            run_state.videos_failed += 1
            return

        # 4. Step succeeded
        state.completed_steps.append(step_name)
        state.updated_at = datetime.now(timezone.utc)
        save_video_state(state, root=state_root)

        # Check kill-switch after every step (cost may have flipped it)
        if run_state.kill_switch_fired:
            log.critical("kill_switch_fired", step=step_name, video_id=video_id)
            ctx.kill_switch_event.set()
            raise KillSwitchFiredError("Kill-switch fired after step completion.")

    # All steps done — preserve completed_local if a step set it (e.g. drive_upload disabled)
    if state.status != "completed_local":
        state.status = "completed"
    state.current_step = None
    state.updated_at = datetime.now(timezone.utc)
    save_video_state(state, root=state_root)
    run_state.videos_completed += 1
    save_run_state(run_state, root=state_root)
    log.info("video_completed", video_id=video_id, product_id=state.product_id)


# ---------------------------------------------------------------------------
# Per-product processing  (SPEC.md §8: videos within a product run serially)
# ---------------------------------------------------------------------------


async def process_product(
    brief: ProductBrief,
    specs: list[VideoSpec],
    ctx: OrchestratorContext,
    run_state: RunState,
) -> None:
    """Process all videos for *brief* serially.

    Videos within one product are processed one at a time (SPEC.md §8).
    """
    for spec in specs:
        # Check kill-switch before admitting each video
        if run_state.kill_switch_fired:
            log.critical("kill_switch_fired", product_id=brief.product_id, video_id=spec.video_id)
            ctx.kill_switch_event.set()
            raise KillSwitchFiredError("Kill-switch fired; refusing to start new video.")

        try:
            await process_video(spec.video_id, brief, ctx, run_state)
        except KillSwitchFiredError:
            ctx.kill_switch_event.set()
            raise


# ---------------------------------------------------------------------------
# Pre-flight config validation
# ---------------------------------------------------------------------------


def _validate_creative_director_config(cfg: dict) -> None:
    """Validate creative_director config values before any Gemini call.

    Raises ``ValueError`` on bad config so that a nonsense ``speaking_clip_index``
    is caught immediately at startup rather than after an expensive LLM call.
    """
    cd_cfg = cfg.get("creative_director", {})
    speaking_idx = int(cd_cfg.get("speaking_clip_index", 1))
    clip_counts_cfg = cd_cfg.get("clip_counts", (2, 3, 2))
    clip_counts = tuple(int(c) for c in clip_counts_cfg)
    for i, cc in enumerate(clip_counts):
        if not (0 <= speaking_idx < cc):
            raise ValueError(
                f"creative_director.speaking_clip_index={speaking_idx} is out of range "
                f"for clip_counts[{i}]={cc}. Must satisfy 0 <= speaking_clip_index < clip_count "
                f"for ALL spec configurations."
            )


# ---------------------------------------------------------------------------
# Top-level pipeline entry point
# ---------------------------------------------------------------------------


async def run_pipeline(
    images: list[Any],  # list[PolledImage]
    ctx: OrchestratorContext,
    run_state: RunState,
    *,
    per_video_max_usd: float,
    global_max_usd: float,
) -> None:
    """Top-level pipeline: analyst → director → per-product video processing.

    Products are processed in parallel up to ``cfg.parallelism.max_concurrent_products``.
    Videos within each product are processed serially (SPEC.md §8).

    After all work completes (or the kill-switch fires), in-flight Veo operations
    are drained (SPEC.md §12).
    """
    from ugc_pipeline.state_manager import (
        _video_state_path,
        load_video_spec,
        load_video_state,
        save_video_spec,
        save_video_state,
    )
    from ugc_pipeline.models import VideoState

    cfg = ctx.cfg

    # Pre-flight: validate creative_director config before any Gemini call.
    _validate_creative_director_config(cfg)

    parallelism_cfg = cfg.get("parallelism", {})
    max_concurrent = int(parallelism_cfg.get("max_concurrent_products", 2))
    product_semaphore = asyncio.Semaphore(max_concurrent)

    sweep_locks(ctx.state_root)

    gemini_client = ctx.effective_gemini_client()

    async def _process_one_image(img: Any) -> None:
        async with product_semaphore:
            # Kill-switch check before admitting
            if run_state.kill_switch_fired:
                ctx.kill_switch_event.set()
                log.critical(
                    "kill_switch_fired",
                    product_id=getattr(img, "product_id", "?"),
                    reason="refused_new_product",
                )
                return

            try:
                admit_product(run_state, img.product_id)
            except KillSwitchFiredError:
                ctx.kill_switch_event.set()
                return

            if img.product_id not in run_state.products_seen:
                run_state.products_seen.append(img.product_id)
                save_run_state(run_state, root=ctx.state_root)

            # Stage 2 — product_analyst
            try:
                naming_cfg = (ctx.brand_guidance.get("naming", {}) or {}) if ctx.brand_guidance else {}
                variant_suffix_pattern = naming_cfg.get("filename_variant_suffix_pattern") or None
                brief = await run_product_analyst(
                    img.image_bytes,
                    img.filename,
                    img.drive_file_id,
                    client=gemini_client,
                    run_state=run_state,
                    state_root=ctx.state_root,
                    global_max_usd=global_max_usd,
                    brand_name=ctx.brand_guidance.get("brand_name", "") if ctx.brand_guidance else "",
                    variant_suffix_pattern=variant_suffix_pattern,
                )
            except Exception as exc:
                log.error("product_analyst_failed", product_id=img.product_id, error=str(exc))
                return

            # Stage 3 — creative_director
            cd_cfg = ctx.cfg.get("creative_director", {})
            clip_counts_cfg = cd_cfg.get("clip_counts")
            speaking_idx_cfg = int(cd_cfg.get("speaking_clip_index", 1))
            cd_kwargs: dict = {"speaking_clip_index": speaking_idx_cfg}
            if clip_counts_cfg:
                cd_kwargs["clip_counts"] = tuple(int(c) for c in clip_counts_cfg)
            try:
                specs = await run_creative_director(
                    brief,
                    ctx.talent_pool,
                    client=gemini_client,
                    run_state=run_state,
                    state_root=ctx.state_root,
                    global_max_usd=global_max_usd,
                    brand_guidance=ctx.brand_guidance or None,
                    **cd_kwargs,
                )
            except Exception as exc:
                log.error("creative_director_failed", product_id=img.product_id, error=str(exc))
                return

            # Initialise VideoState files for any newly created specs
            for spec in specs:
                state_path = _video_state_path(spec.video_id, ctx.state_root)
                if not state_path.exists():
                    vs = VideoState(
                        video_id=spec.video_id,
                        product_id=spec.product_id,
                        spec_index=spec.spec_index,
                    )
                    save_video_state(vs, root=ctx.state_root)

            # Stages 4–11 — per-video processing (serially within product)
            try:
                await process_product(brief, specs, ctx, run_state)
            except KillSwitchFiredError:
                ctx.kill_switch_event.set()

    tasks = [_process_one_image(img) for img in images]
    await asyncio.gather(*tasks)

    # Drain in-flight Veo operations after all product tasks complete
    if ctx.in_flight_veo:
        veo_cfg = cfg.get("veo", {})
        poll_interval = float(veo_cfg.get("poll_interval_seconds", 15.0))
        poll_timeout = float(veo_cfg.get("poll_timeout_seconds", 360.0))

        # Build a video_states_by_video_id map from state files
        from ugc_pipeline.state_manager import load_video_state

        video_states: dict[str, Any] = {}
        for op in ctx.in_flight_veo:
            if op.video_id and op.video_id not in video_states:
                try:
                    video_states[op.video_id] = load_video_state(op.video_id, ctx.state_root)
                except FileNotFoundError:
                    pass

        await drain_inflight_veo(
            ctx.in_flight_veo,
            ctx.veo_client,
            run_state,
            video_states,
            ctx.state_root,
            ctx.artifacts_root,
            poll_interval,
            poll_timeout,
        )

    # Finalise run state
    run_state.ended_at = datetime.now(timezone.utc)
    save_run_state(run_state, root=ctx.state_root)

    if run_state.kill_switch_fired:
        log.critical(
            "pipeline_ended_kill_switch",
            cumulative_cost_usd=run_state.cumulative_cost_usd,
            global_max_usd=global_max_usd,
        )
        raise KillSwitchFiredError(
            f"Global budget cap of ${global_max_usd:.2f} reached. "
            f"Cumulative cost this run: ${run_state.cumulative_cost_usd:.2f}"
        )

    log.info(
        "pipeline_completed",
        videos_completed=run_state.videos_completed,
        videos_failed=run_state.videos_failed,
        cumulative_cost_usd=run_state.cumulative_cost_usd,
    )
