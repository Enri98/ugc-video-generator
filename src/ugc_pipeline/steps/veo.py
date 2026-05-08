"""Veo generation step for the UGC pipeline.

Implements SPEC.md §5 Steps 5 (veo_generation) and 6 (safety_retry).

Cost model (SPEC.md §12):
  - Veo 3.1 Fast: ~$0.30–$0.50 per clip (5–8 s); midpoint $0.40 used as the estimate.
  - Safety retry (Gemini Flash rewrite): ~$0.001 flat.

Cost is recorded BEFORE each API call attempt so a crash during generation still
registers in cost accounting on the next run (SPEC.md §12 "bill on submit" rule).
"""

from __future__ import annotations

import asyncio
import hashlib
import pathlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import structlog
import tenacity

from ugc_pipeline.cost_tracker import increment_cost
from ugc_pipeline.models import ProductBrief, PromptVersion, RunState, VideoSpec, VideoState
from ugc_pipeline.prompts import safety_retry as safety_retry_prompt
from ugc_pipeline.state_manager import (
    _dump_model,
    _run_state_path,
    _video_state_path,
    save_run_state,
    save_video_state,
    write_state_atomic,
)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions  (SPEC.md §11)
# ---------------------------------------------------------------------------


class VeoError(Exception):
    """Base error for Veo 3.1 Fast failures."""


class VeoSafetyBlockError(VeoError):
    """Raised when Veo rejects a generation request due to a safety filter."""


class VeoTimeoutError(VeoError):
    """Raised when the Veo operation does not complete within the allowed poll window."""


class VeoGenerationError(VeoError):
    """Raised for non-safety, non-timeout Veo generation failures."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VeoOperation:
    """Tracks an in-flight Veo long-running operation."""

    operation_id: str
    clip_index: int
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    video_id: str = ""  # video_id of the parent VideoState; set when submitting


@dataclass
class VeoResult:
    """Container for a successful Veo clip generation result."""

    mp4_bytes: bytes
    operation_id: str
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Client protocols
# ---------------------------------------------------------------------------


class VeoClientProtocol(Protocol):
    """Minimal async interface for the Veo 3.1 Fast image-to-video API."""

    async def submit(
        self,
        *,
        image_bytes: bytes,
        prompt: str,
        model: str,
    ) -> str:
        """Submit a clip generation request.

        Returns the operation_id for the long-running operation.
        """
        ...

    async def poll(self, operation_id: str) -> dict:
        """Poll the status of a long-running Veo operation.

        Returns a dict with keys:
            ``done`` (bool): True if the operation has finished.
            ``mp4_bytes`` (bytes | None): Raw MP4 bytes on success; None otherwise.
            ``error`` (str | None): Error message if the operation failed.
            ``safety_block`` (bool): True if blocked by a safety filter.
        """
        ...


class GeminiFlashClientProtocol(Protocol):
    """Minimal async interface for the Gemini Flash safety-retry rewrite call."""

    async def rewrite(self, prompt: str) -> str:
        """Rewrite a blocked scene prompt using Gemini Flash (plain text response).

        Returns the rewritten prompt string.
        """
        ...


# ---------------------------------------------------------------------------
# Cost estimates  (SPEC.md §12)
# ---------------------------------------------------------------------------

_VEO_CLIP_COST_USD: float = 0.40   # midpoint of $0.30–$0.50 range for 5–8 s clip
_SAFETY_RETRY_COST_USD: float = 0.001  # Gemini Flash call for one rewrite


def estimate_veo_clip_cost_usd(clip_seconds: float = 6.0) -> float:
    """Return the per-clip cost estimate for a Veo 3.1 Fast generation.

    Uses a flat midpoint of $0.40 regardless of *clip_seconds* in v1. The range
    documented in SPEC.md §12 is $0.30–$0.50 for 5–8 second clips; tune after
    Day 8 empirical billing data is available.
    """
    return _VEO_CLIP_COST_USD


def estimate_safety_retry_rewrite_cost_usd() -> float:
    """Return the flat cost estimate for one Gemini Flash safety-retry rewrite call."""
    return _SAFETY_RETRY_COST_USD


# ---------------------------------------------------------------------------
# Low-level API wrappers with tenacity
# ---------------------------------------------------------------------------


async def submit_veo_operation(
    client: VeoClientProtocol,
    *,
    image_bytes: bytes,
    prompt: str,
    model: str = "veo-3.1-fast",
) -> str:
    """Submit an image-to-video operation to Veo and return the operation_id.

    Wraps ``client.submit`` with tenacity (up to 3 attempts) for transient errors.
    Note: Re-submission on transient error may cause double-billing if the first
    request was accepted but the response was lost. This risk is documented in
    SPEC.md §11 and accepted in v1. Safety blocks and generation errors are not
    retried here — they are handled at the caller level.
    """

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=2, max=30),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _submit() -> str:
        return await client.submit(image_bytes=image_bytes, prompt=prompt, model=model)

    return await _submit()


async def poll_veo_operation(
    client: VeoClientProtocol,
    operation_id: str,
    *,
    poll_interval_seconds: float,
    poll_timeout_seconds: float,
) -> VeoResult:
    """Poll a Veo long-running operation until done, timed out, or error.

    Parameters
    ----------
    client:
        Veo client satisfying VeoClientProtocol.
    operation_id:
        The operation ID returned by ``submit_veo_operation``.
    poll_interval_seconds:
        Time to wait between poll attempts.
    poll_timeout_seconds:
        Maximum total wall-clock time before raising VeoTimeoutError.

    Raises
    ------
    VeoSafetyBlockError:
        If the operation ended with a safety block.
    VeoGenerationError:
        If the operation ended with a non-safety error.
    VeoTimeoutError:
        If the operation did not complete within *poll_timeout_seconds*.
    """
    deadline = time.monotonic() + poll_timeout_seconds

    while True:
        result = await client.poll(operation_id)

        if result.get("done"):
            if result.get("safety_block"):
                raise VeoSafetyBlockError(
                    f"Veo operation {operation_id!r} was blocked by a safety filter."
                )
            error_msg = result.get("error")
            if error_msg:
                raise VeoGenerationError(
                    f"Veo operation {operation_id!r} failed: {error_msg}"
                )
            mp4_bytes = result.get("mp4_bytes")
            if mp4_bytes:
                return VeoResult(mp4_bytes=mp4_bytes, operation_id=operation_id)
            # done=True but no bytes and no error — treat as a generation error
            raise VeoGenerationError(
                f"Veo operation {operation_id!r} reported done but returned no bytes."
            )

        if time.monotonic() >= deadline:
            raise VeoTimeoutError(
                f"Veo operation {operation_id!r} did not complete within "
                f"{poll_timeout_seconds:.1f} seconds."
            )

        await asyncio.sleep(poll_interval_seconds)


# ---------------------------------------------------------------------------
# Safety retry sub-step  (SPEC.md §5 Step 6)
# ---------------------------------------------------------------------------


async def run_safety_retry_for_clip(
    spec: VideoSpec,
    clip_index: int,
    *,
    flash_client: GeminiFlashClientProtocol,
    video_state: VideoState,
    run_state: RunState,
    state_root: pathlib.Path,
    global_max_usd: float,
) -> str:
    """Rewrite the blocked scene prompt via Gemini Flash and return the new prompt.

    Bills ``safety_retry_usd`` BEFORE the Gemini Flash call (SPEC.md §12).

    Parameters
    ----------
    spec:
        VideoSpec containing the original scene descriptions.
    clip_index:
        0-based index of the clip being retried.
    flash_client:
        Gemini Flash client satisfying GeminiFlashClientProtocol.
    video_state:
        Mutable VideoState; mutated and saved after cost increment.
    run_state:
        Mutable RunState; mutated and saved after cost increment.
    state_root:
        Root of the state directory tree.
    global_max_usd:
        Kill-switch threshold forwarded to increment_cost.

    Returns
    -------
    str
        The rewritten scene prompt.
    """
    video_state_path = _video_state_path(video_state.video_id, state_root)
    run_state_path = _run_state_path(run_state.run_id, state_root)

    original_prompt = spec.scene_descriptions[clip_index]
    rendered_prompt = safety_retry_prompt.render(original_prompt=original_prompt)

    prompt_sha = hashlib.sha256(rendered_prompt.encode()).hexdigest()
    prompt_version = PromptVersion(
        step_name="safety_retry",
        version=safety_retry_prompt.VERSION,
        content_sha256=prompt_sha,
        rendered_at=datetime.now(timezone.utc),
    )
    video_state.prompt_versions["safety_retry"] = prompt_version

    # Bill BEFORE the call
    increment_cost(
        run_state,
        video_state,
        "safety_retry_usd",
        estimate_safety_retry_rewrite_cost_usd(),
        global_max_usd,
    )
    write_state_atomic(video_state_path, _dump_model(video_state))
    write_state_atomic(run_state_path, _dump_model(run_state))

    log.info(
        "veo_safety_retry_rewrite_started",
        video_id=video_state.video_id,
        clip_index=clip_index,
    )

    new_prompt = await flash_client.rewrite(rendered_prompt)

    log.info(
        "veo_safety_retry_rewrite_done",
        video_id=video_state.video_id,
        clip_index=clip_index,
    )

    return new_prompt


# ---------------------------------------------------------------------------
# Main per-clip entry point  (SPEC.md §5 Steps 5 + 6)
# ---------------------------------------------------------------------------


async def run_veo_for_clip(
    spec: VideoSpec,
    brief: ProductBrief,
    clip_index: int,
    *,
    client: VeoClientProtocol,
    flash_client: GeminiFlashClientProtocol | None,
    video_state: VideoState,
    run_state: RunState,
    state_root: pathlib.Path,
    artifacts_root: pathlib.Path,
    poll_interval_seconds: float = 15.0,
    poll_timeout_seconds: float = 360.0,
    global_max_usd: float = 50.0,
    dry_run: bool = False,
) -> pathlib.Path:
    """Generate a raw MP4 clip for *clip_index* and return its local path.

    This function implements both Step 5 (veo_generation) and Step 6
    (safety_retry) per SPEC.md §5.

    Key behaviours:
    - Idempotency: if the artifact already exists on disk, the step is skipped.
    - Resume: if an operation_id is already stored in state (key ``op:clip_{i}``)
      but no artifact exists, polling resumes from the stored ID without re-submitting.
    - Dry-run: no API calls; logs the prompt and estimated cost.
    - Cost is recorded BEFORE each Veo submit (SPEC.md §12 "bill on submit" rule).
    - Safety retry: on VeoSafetyBlockError, if *flash_client* is provided, the
      scene prompt is rewritten and re-submitted once. A second block is terminal.

    Parameters
    ----------
    spec:
        VideoSpec for the current video.
    brief:
        ProductBrief describing the product (used for first-frame lookup).
    clip_index:
        0-based index of the clip being processed.
    client:
        VeoClientProtocol implementation (real or mock).
    flash_client:
        GeminiFlashClientProtocol implementation for safety retry, or None to
        disable safety retry.
    video_state:
        Mutable VideoState for this video; mutated in place and saved.
    run_state:
        Mutable RunState for this run; mutated in place and saved.
    state_root:
        Root of the state directory tree.
    artifacts_root:
        Root of the artifacts directory.
    poll_interval_seconds:
        Seconds between Veo poll attempts.
    poll_timeout_seconds:
        Maximum total wall-clock seconds to wait for a Veo operation.
    global_max_usd:
        Kill-switch threshold forwarded to increment_cost.
    dry_run:
        If True, skip all API calls, log the prompt and estimated cost, and
        return the expected output path without creating it.

    Returns
    -------
    pathlib.Path
        Path to the written raw MP4 file.
    """
    t_start = time.monotonic()

    artifact_key = f"clip_{clip_index}_raw"
    op_key = f"op:clip_{clip_index}"
    out_path = artifacts_root / video_state.video_id / f"clip_{clip_index}_raw.mp4"

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
                step="veo_generation",
                video_id=video_state.video_id,
                clip_index=clip_index,
                artifact_key=artifact_key,
            )
            return existing_path

    # ------------------------------------------------------------------
    # 2. Dry-run short-circuit
    # ------------------------------------------------------------------
    scene_prompt = spec.scene_descriptions[clip_index]

    if dry_run:
        log.info(
            "step_skipped_dryrun",
            step="veo_generation",
            video_id=video_state.video_id,
            clip_index=clip_index,
            estimated_cost_usd=estimate_veo_clip_cost_usd(),
            scene_prompt=scene_prompt,
        )
        return out_path

    log.info(
        "step_started",
        step="veo_generation",
        video_id=video_state.video_id,
        clip_index=clip_index,
    )

    # ------------------------------------------------------------------
    # 3. Read first-frame PNG bytes from artifact store
    # ------------------------------------------------------------------
    firstframe_key = f"clip_{clip_index}_firstframe"
    firstframe_path_str = video_state.artifacts.get(firstframe_key)
    if firstframe_path_str is None:
        raise VeoGenerationError(
            f"First-frame artifact {firstframe_key!r} not found in VideoState.artifacts "
            f"for video {video_state.video_id!r}. Run first_frame step first."
        )
    firstframe_path = pathlib.Path(firstframe_path_str)
    image_bytes = firstframe_path.read_bytes()

    # ------------------------------------------------------------------
    # 4. Helper: submit + poll one attempt (handles cost accounting)
    # ------------------------------------------------------------------
    async def _submit_and_poll(prompt: str) -> VeoResult:
        """Submit one Veo operation, bill, persist state, and poll to completion."""
        # Check whether we already have an in-flight operation to resume
        stored_op_id = video_state.artifacts.get(op_key)

        if stored_op_id is None:
            # Bill BEFORE the submit call (SPEC.md §12)
            increment_cost(
                run_state,
                video_state,
                "veo_usd",
                estimate_veo_clip_cost_usd(),
                global_max_usd,
            )
            write_state_atomic(video_state_path, _dump_model(video_state))
            write_state_atomic(run_state_path, _dump_model(run_state))

            operation_id = await submit_veo_operation(
                client,
                image_bytes=image_bytes,
                prompt=prompt,
            )
            # Store operation_id so a crash during poll allows resume
            video_state.artifacts[op_key] = operation_id
            write_state_atomic(video_state_path, _dump_model(video_state))

            log.info(
                "veo_operation_submitted",
                video_id=video_state.video_id,
                clip_index=clip_index,
                operation_id=operation_id,
            )
        else:
            operation_id = stored_op_id
            log.info(
                "veo_operation_resumed",
                video_id=video_state.video_id,
                clip_index=clip_index,
                operation_id=operation_id,
            )

        return await poll_veo_operation(
            client,
            operation_id,
            poll_interval_seconds=poll_interval_seconds,
            poll_timeout_seconds=poll_timeout_seconds,
        )

    # ------------------------------------------------------------------
    # 5. First attempt
    # ------------------------------------------------------------------
    try:
        veo_result = await _submit_and_poll(scene_prompt)
    except VeoSafetyBlockError:
        log.warning(
            "veo_safety_block",
            video_id=video_state.video_id,
            clip_index=clip_index,
        )
        if flash_client is None:
            raise

        # Safety retry (Step 6): rewrite and re-submit once
        rewritten_prompt = await run_safety_retry_for_clip(
            spec,
            clip_index,
            flash_client=flash_client,
            video_state=video_state,
            run_state=run_state,
            state_root=state_root,
            global_max_usd=global_max_usd,
        )

        # Clear the stored op_key so _submit_and_poll does a fresh submit
        video_state.artifacts.pop(op_key, None)

        # Bill the second Veo submit BEFORE calling
        increment_cost(
            run_state,
            video_state,
            "veo_usd",
            estimate_veo_clip_cost_usd(),
            global_max_usd,
        )
        write_state_atomic(video_state_path, _dump_model(video_state))
        write_state_atomic(run_state_path, _dump_model(run_state))

        second_op_id = await submit_veo_operation(
            client,
            image_bytes=image_bytes,
            prompt=rewritten_prompt,
        )
        video_state.artifacts[op_key] = second_op_id
        write_state_atomic(video_state_path, _dump_model(video_state))

        log.info(
            "veo_operation_submitted",
            video_id=video_state.video_id,
            clip_index=clip_index,
            operation_id=second_op_id,
            attempt=2,
        )

        # This will raise VeoSafetyBlockError again if still blocked (terminal)
        veo_result = await poll_veo_operation(
            client,
            second_op_id,
            poll_interval_seconds=poll_interval_seconds,
            poll_timeout_seconds=poll_timeout_seconds,
        )

    # ------------------------------------------------------------------
    # 6. Persist artifact
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(veo_result.mp4_bytes)

    video_state.artifacts[artifact_key] = str(out_path)
    video_state.updated_at = datetime.now(timezone.utc)
    save_video_state(video_state, root=state_root)
    save_run_state(run_state, root=state_root)

    duration_ms = int((time.monotonic() - t_start) * 1000)
    log.info(
        "veo_operation_done",
        video_id=video_state.video_id,
        clip_index=clip_index,
        artifact_path=str(out_path),
        duration_ms=duration_ms,
        operation_id=veo_result.operation_id,
        cost_usd_cumulative=run_state.cumulative_cost_usd,
    )

    return out_path


# ---------------------------------------------------------------------------
# Kill-switch drain helper  (SPEC.md §12 + §13)
# ---------------------------------------------------------------------------


async def drain_inflight_veo(
    operations: list[VeoOperation],
    client: VeoClientProtocol,
    run_state: RunState,
    video_states_by_video_id: dict[str, VideoState],
    state_root: pathlib.Path,
    artifacts_root: pathlib.Path,
    poll_interval_seconds: float,
    poll_timeout_seconds: float,
) -> None:
    """Poll all in-flight Veo operations to completion after the kill-switch fires.

    No new work is submitted. Each operation is polled until it finishes
    (success or error) or times out. Any produced MP4 bytes are written to disk
    and the corresponding VideoState artifact is updated.

    This function satisfies the SPEC.md §12 graceful kill-switch drain contract:
    "drain in-flight paid operations before halting".

    Parameters
    ----------
    operations:
        List of VeoOperation instances representing in-flight operations.
    client:
        VeoClientProtocol for polling.
    run_state:
        Mutable RunState for cost tracking; saved after each artifact write.
    video_states_by_video_id:
        Mapping of video_id -> VideoState for each video with in-flight ops.
    state_root:
        Root of the state directory tree.
    artifacts_root:
        Root of the artifacts directory.
    poll_interval_seconds:
        Seconds between poll attempts.
    poll_timeout_seconds:
        Maximum total wall-clock seconds to wait per operation.
    """
    for op in operations:
        video_state = video_states_by_video_id.get(op.video_id)
        try:
            veo_result = await poll_veo_operation(
                client,
                op.operation_id,
                poll_interval_seconds=poll_interval_seconds,
                poll_timeout_seconds=poll_timeout_seconds,
            )

            # Determine the expected artifact path for this clip
            if video_state is not None:
                out_path = artifacts_root / op.video_id / f"clip_{op.clip_index}_raw.mp4"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(veo_result.mp4_bytes)

                artifact_key = f"clip_{op.clip_index}_raw"
                video_state.artifacts[artifact_key] = str(out_path)
                video_state.updated_at = datetime.now(timezone.utc)
                save_video_state(video_state, root=state_root)
                save_run_state(run_state, root=state_root)

                log.info(
                    "drain_inflight_veo_done",
                    operation_id=op.operation_id,
                    clip_index=op.clip_index,
                    artifact_path=str(out_path),
                )

        except (VeoSafetyBlockError, VeoTimeoutError, VeoGenerationError) as exc:
            log.warning(
                "drain_inflight_veo_failed",
                operation_id=op.operation_id,
                clip_index=op.clip_index,
                error=str(exc),
            )
            # Continue draining remaining operations even if one fails
            continue
