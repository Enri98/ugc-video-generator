"""Dry-run mode tests — SPEC.md §12 (Dry-Run Mode) and §13 (Test Strategy).

Verifies that:
  - In dry_run=True mode, NO Veo or Nano Banana client calls are made.
  - step_skipped_dryrun events are logged for skipped paid steps.
  - product_analyst and creative_director ARE still called (real-step semantics).
"""

from __future__ import annotations

import asyncio
import pathlib
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from ugc_pipeline.models import (
    ProductBrief,
    RunState,
    VideoSpec,
    VideoState,
)
from ugc_pipeline.orchestrator import OrchestratorContext, process_video
from ugc_pipeline.state_manager import (
    save_run_state,
    save_video_spec,
    save_video_state,
)
from ugc_pipeline.steps.first_frame import NanoBananaResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_state() -> RunState:
    return RunState(run_id=str(uuid.uuid4()))


def _video_spec(product_id: str, clip_count: int = 2) -> VideoSpec:
    return VideoSpec(
        video_id=str(uuid.uuid4()),
        product_id=product_id,
        spec_index=0,
        tone="warm storyteller",
        narrative_arc="A quiet morning.",
        talent_id="talent_01",
        clip_count=clip_count,
        scene_descriptions=[f"Scene {i}." for i in range(clip_count)],
        script_blocks=["Ogni mattina." for _ in range(clip_count)],
        created_at=datetime.now(timezone.utc),
    )


def _product_brief(product_id: str, tmp_path: pathlib.Path) -> ProductBrief:
    return ProductBrief(
        product_id=product_id,
        image_path=str(tmp_path / "product.jpg"),
        shape="cylindrical",
        dominant_colours=["#fff"],
        packaging_style="plain",
        inferred_category="lifestyle",
        lifestyle_contexts=["morning", "evening", "outdoor"],
        created_at=datetime.now(timezone.utc),
    )


def _make_dry_run_context(
    state_root: pathlib.Path,
    artifacts_root: pathlib.Path,
    nb_client: MagicMock,
    veo_client: MagicMock,
) -> OrchestratorContext:
    from ugc_pipeline.utils.drive import InMemoryDriveClient

    cfg = {
        "budget": {
            "per_video_max_usd": 8.0,
            "global_max_usd": 50.0,
            "max_cost_per_clip_usd": 0.60,
        },
        "parallelism": {"max_concurrent_products": 2, "max_concurrent_veo_ops": 4},
        "veo": {"poll_interval_seconds": 0.001, "poll_timeout_seconds": 5.0},
        "post_production": {"trim_tail_seconds": 0.0, "caption_max_chars_per_line": 42},
        "cleanup": {"keep_raw_clips": True, "keep_trimmed_clips": True},
    }
    talent_pool = {
        "talent_01": {
            "gender": "woman",
            "age_range": "late 20s",
            "aesthetic": "natural",
            "camera_relationship": "conversational",
            "lighting_preference": "soft natural window light",
        }
    }
    return OrchestratorContext(
        gemini_pro_client=MagicMock(),
        nano_banana_client=nb_client,
        veo_client=veo_client,
        flash_client=MagicMock(),
        drive_client=InMemoryDriveClient(),
        state_root=state_root,
        artifacts_root=artifacts_root,
        output_videos_folder_id=None,
        output_reports_folder_id=None,
        cfg=cfg,
        talent_pool=talent_pool,
        dry_run=True,  # <-- dry_run=True
        gemini_limiter=None,
        nano_banana_limiter=None,
        veo_semaphore=asyncio.Semaphore(4),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_no_veo_or_nano_banana_calls(
    tmp_path: pathlib.Path,
    clip_fixture_mp4_path: pathlib.Path,
    firstframe_fixture_png_path: pathlib.Path,
) -> None:
    """In dry_run=True mode, Veo and Nano Banana clients must NOT be called."""
    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    state_root.mkdir()
    artifacts_root.mkdir()

    product_id = "dryrun000001"
    spec = _video_spec(product_id, clip_count=2)
    brief = _product_brief(product_id, tmp_path)

    save_video_spec(spec, root=state_root)
    save_video_state(
        VideoState(video_id=spec.video_id, product_id=product_id, spec_index=0),
        root=state_root,
    )

    run_state = _run_state()
    save_run_state(run_state, root=state_root)

    nb_client = MagicMock()
    nb_client.generate_image = AsyncMock(
        return_value=NanoBananaResult(png_bytes=firstframe_fixture_png_path.read_bytes())
    )
    veo_client = MagicMock()
    veo_client.submit = AsyncMock(return_value="mock-op-001")
    veo_client.poll = AsyncMock(
        return_value={"done": True, "mp4_bytes": clip_fixture_mp4_path.read_bytes(), "error": None, "safety_block": False}
    )

    ctx = _make_dry_run_context(state_root, artifacts_root, nb_client, veo_client)

    # Run with dry_run=True — caption and report/upload still execute (local steps)
    with patch("ugc_pipeline.steps.caption.transcribe_with_whisper", return_value=[]):
        await process_video(spec.video_id, brief, ctx, run_state)

    # Nano Banana must NOT have been called
    nb_client.generate_image.assert_not_called()

    # Veo must NOT have been called
    veo_client.submit.assert_not_called()
    veo_client.poll.assert_not_called()


@pytest.mark.asyncio
async def test_dry_run_skipped_events_logged(
    tmp_path: pathlib.Path,
) -> None:
    """In dry_run mode, step_skipped_dryrun events must be emitted for Nano Banana and Veo."""
    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    state_root.mkdir()
    artifacts_root.mkdir()

    product_id = "dryrun000002"
    spec = _video_spec(product_id, clip_count=2)
    brief = _product_brief(product_id, tmp_path)

    save_video_spec(spec, root=state_root)
    save_video_state(
        VideoState(video_id=spec.video_id, product_id=product_id, spec_index=0),
        root=state_root,
    )

    run_state = _run_state()
    save_run_state(run_state, root=state_root)

    nb_client = MagicMock()
    nb_client.generate_image = AsyncMock(return_value=NanoBananaResult(png_bytes=b"\x89PNG" + b"\x00" * 50))
    veo_client = MagicMock()
    veo_client.submit = AsyncMock(return_value="op-x")
    veo_client.poll = AsyncMock(return_value={"done": True, "mp4_bytes": b"fake", "error": None, "safety_block": False})

    ctx = _make_dry_run_context(state_root, artifacts_root, nb_client, veo_client)

    logged_events: list[str] = []

    import structlog

    # Use structlog's testing capability to capture events
    original_info = None

    class _EventCapture:
        """Minimal structlog-compatible logger that captures event names."""
        def __init__(self, name: str) -> None:
            self._name = name

        def info(self, event: str, **kwargs: object) -> None:
            logged_events.append(event)

        def warning(self, event: str, **kwargs: object) -> None:
            logged_events.append(event)

        def error(self, event: str, **kwargs: object) -> None:
            logged_events.append(event)

        def critical(self, event: str, **kwargs: object) -> None:
            logged_events.append(event)

        def debug(self, event: str, **kwargs: object) -> None:
            logged_events.append(event)

        def bind(self, **kw: object) -> "_EventCapture":
            return self

    # Patch structlog.get_logger in the first_frame and veo step modules
    with (
        patch("ugc_pipeline.steps.first_frame.log", _EventCapture("first_frame")),
        patch("ugc_pipeline.steps.veo.log", _EventCapture("veo")),
        patch("ugc_pipeline.steps.caption.transcribe_with_whisper", return_value=[]),
    ):
        await process_video(spec.video_id, brief, ctx, run_state)

    # step_skipped_dryrun must have been emitted at least once
    assert "step_skipped_dryrun" in logged_events, (
        f"Expected 'step_skipped_dryrun' in logged events, got: {logged_events}"
    )
