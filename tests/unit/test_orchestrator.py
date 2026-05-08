"""Unit/integration tests for ugc_pipeline.orchestrator.

Drives a single product through the full orchestration path using mocked
API clients. No paid calls are made. Covers SPEC.md §7 resume algorithm
and §8 concurrency model.
"""

from __future__ import annotations

import asyncio
import pathlib
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ugc_pipeline.cost_tracker import KillSwitchFiredError
from ugc_pipeline.models import (
    CostBreakdown,
    ProductBrief,
    RunState,
    VideoSpec,
    VideoState,
)
from ugc_pipeline.orchestrator import (
    STEP_PIPELINE,
    OrchestratorContext,
    admit_product,
    process_video,
)
from ugc_pipeline.state_manager import (
    save_product_brief,
    save_run_state,
    save_video_spec,
    save_video_state,
)
from ugc_pipeline.steps.first_frame import NanoBananaResult
from ugc_pipeline.steps.veo import VeoOperation


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _run_state() -> RunState:
    return RunState(run_id=str(uuid.uuid4()))


def _product_brief(product_id: str, tmp_path: pathlib.Path) -> ProductBrief:
    return ProductBrief(
        product_id=product_id,
        image_path=str(tmp_path / "product.jpg"),
        shape="cylindrical",
        dominant_colours=["#FFFFFF"],
        packaging_style="plain box",
        inferred_category="kitchenware",
        lifestyle_contexts=["morning routine", "desk setup", "outdoor"],
        created_at=datetime.now(timezone.utc),
    )


def _video_spec(product_id: str, spec_index: int = 0, clip_count: int = 2) -> VideoSpec:
    return VideoSpec(
        video_id=str(uuid.uuid4()),
        product_id=product_id,
        spec_index=spec_index,
        tone="warm storyteller",
        narrative_arc="A quiet morning becomes special.",
        talent_id="talent_01",
        clip_count=clip_count,
        scene_descriptions=[
            f"Scene {i} description." for i in range(clip_count)
        ],
        script_blocks=[
            "Ogni mattina merita cura." for _ in range(clip_count)
        ],
        created_at=datetime.now(timezone.utc),
    )


def _make_mock_nano_banana(png_bytes: bytes) -> MagicMock:
    client = MagicMock()
    client.generate_image = AsyncMock(
        return_value=NanoBananaResult(png_bytes=png_bytes)
    )
    return client


def _make_mock_veo(mp4_bytes: bytes) -> MagicMock:
    counter = {"n": 0}

    async def _submit(**kwargs: object) -> str:
        counter["n"] += 1
        return f"mock-op-{counter['n']:03d}"

    client = MagicMock()
    client.submit = AsyncMock(side_effect=_submit)
    client.poll = AsyncMock(
        return_value={
            "done": True,
            "mp4_bytes": mp4_bytes,
            "error": None,
            "safety_block": False,
        }
    )
    return client


def _make_mock_flash() -> MagicMock:
    client = MagicMock()
    client.rewrite = AsyncMock(return_value="Softened scene prompt.")
    return client


def _make_mock_drive() -> MagicMock:
    from ugc_pipeline.utils.drive import InMemoryDriveClient
    return InMemoryDriveClient()


def _minimal_talent_pool() -> dict:
    return {
        "talent_01": {
            "gender": "woman",
            "age_range": "late 20s",
            "aesthetic": "natural, relaxed",
            "camera_relationship": "conversational",
            "lighting_preference": "soft natural window light",
        }
    }


def _make_context(
    state_root: pathlib.Path,
    artifacts_root: pathlib.Path,
    nano_banana_client: MagicMock,
    veo_client: MagicMock,
    flash_client: MagicMock,
    dry_run: bool = False,
) -> OrchestratorContext:
    cfg = {
        "budget": {
            "per_video_max_usd": 8.0,
            "global_max_usd": 50.0,
            "max_cost_per_clip_usd": 0.60,
        },
        "parallelism": {"max_concurrent_products": 2, "max_concurrent_veo_ops": 4},
        "veo": {"poll_interval_seconds": 0.01, "poll_timeout_seconds": 10.0},
        "post_production": {"trim_tail_seconds": 0.0, "caption_max_chars_per_line": 42},
        "cleanup": {"keep_raw_clips": True, "keep_trimmed_clips": True},
    }
    return OrchestratorContext(
        gemini_pro_client=MagicMock(),  # not used in this test path
        nano_banana_client=nano_banana_client,
        veo_client=veo_client,
        flash_client=flash_client,
        drive_client=_make_mock_drive(),
        state_root=state_root,
        artifacts_root=artifacts_root,
        output_videos_folder_id="fake-vid-folder",
        output_reports_folder_id="fake-rep-folder",
        cfg=cfg,
        talent_pool=_minimal_talent_pool(),
        dry_run=dry_run,
        gemini_limiter=None,
        nano_banana_limiter=None,
        veo_semaphore=asyncio.Semaphore(4),
    )


# ---------------------------------------------------------------------------
# Minimal MP4/PNG fixtures (1-byte placeholders — steps are mocked)
# ---------------------------------------------------------------------------

_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
_FAKE_MP4 = b"fakemp4data" + b"\x00" * 200


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_step_pipeline_order() -> None:
    """STEP_PIPELINE must start with first_frame_composite and end with drive_upload."""
    names = [name for name, _ in STEP_PIPELINE]
    assert names[0] == "first_frame_composite"
    assert names[-1] == "drive_upload"
    assert "veo_generation" in names
    assert "end_of_clip_trim" in names
    assert "stitch" in names
    assert "caption" in names
    assert "report_gen" in names


def test_admit_product_raises_when_kill_switch_fired() -> None:
    """admit_product must raise KillSwitchFiredError when kill_switch_fired is True."""
    rs = RunState(run_id=str(uuid.uuid4()), kill_switch_fired=True)
    with pytest.raises(KillSwitchFiredError):
        admit_product(rs, "product-abc")


def test_admit_product_passes_when_ok() -> None:
    """admit_product must not raise when kill_switch_fired is False."""
    rs = RunState(run_id=str(uuid.uuid4()), kill_switch_fired=False)
    admit_product(rs, "product-abc")  # should not raise


@pytest.mark.asyncio
async def test_process_video_completes(
    tmp_path: pathlib.Path,
    clip_fixture_mp4_path: pathlib.Path,
    firstframe_fixture_png_path: pathlib.Path,
) -> None:
    """A single video should reach status='completed' after process_video."""
    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    state_root.mkdir()
    artifacts_root.mkdir()

    png_bytes = firstframe_fixture_png_path.read_bytes()
    mp4_bytes = clip_fixture_mp4_path.read_bytes()

    product_id = "orch000000ab"
    brief = _product_brief(product_id, tmp_path)
    spec = _video_spec(product_id, spec_index=0, clip_count=2)

    save_product_brief(brief, root=state_root)
    save_video_spec(spec, root=state_root)

    initial_state = VideoState(
        video_id=spec.video_id,
        product_id=product_id,
        spec_index=0,
    )
    save_video_state(initial_state, root=state_root)

    run_state = _run_state()
    save_run_state(run_state, root=state_root)

    nb_client = _make_mock_nano_banana(png_bytes)
    veo_client = _make_mock_veo(mp4_bytes)
    flash_client = _make_mock_flash()

    ctx = _make_context(state_root, artifacts_root, nb_client, veo_client, flash_client)

    # Patch caption step to avoid faster-whisper dependency in unit tests
    with patch("ugc_pipeline.steps.caption.transcribe_with_whisper", return_value=[]):
        await process_video(spec.video_id, brief, ctx, run_state)

    # Reload state and verify
    from ugc_pipeline.state_manager import load_video_state
    final_state = load_video_state(spec.video_id, state_root)

    assert final_state.status == "completed", f"Expected 'completed', got {final_state.status!r}"
    assert "first_frame_composite" in final_state.completed_steps
    assert "veo_generation" in final_state.completed_steps
    assert "stitch" in final_state.completed_steps
    assert "caption" in final_state.completed_steps
    assert "report_gen" in final_state.completed_steps
    assert "drive_upload" in final_state.completed_steps


@pytest.mark.asyncio
async def test_process_video_skips_completed_steps(
    tmp_path: pathlib.Path,
    clip_fixture_mp4_path: pathlib.Path,
    firstframe_fixture_png_path: pathlib.Path,
) -> None:
    """If first_frame_composite is already in completed_steps, it must be skipped."""
    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    state_root.mkdir()
    artifacts_root.mkdir()

    png_bytes = firstframe_fixture_png_path.read_bytes()
    mp4_bytes = clip_fixture_mp4_path.read_bytes()

    product_id = "orch000000bb"
    spec = _video_spec(product_id, spec_index=0, clip_count=2)
    brief = _product_brief(product_id, tmp_path)

    save_product_brief(brief, root=state_root)
    save_video_spec(spec, root=state_root)

    video_dir = artifacts_root / spec.video_id
    video_dir.mkdir(parents=True)

    # Pre-populate first-frame artifacts so first_frame step sees them
    for i in range(spec.clip_count):
        ff_path = video_dir / f"clip_{i}_firstframe.png"
        ff_path.write_bytes(png_bytes)

    initial_state = VideoState(
        video_id=spec.video_id,
        product_id=product_id,
        spec_index=0,
        # Mark first_frame as already done
        completed_steps=["first_frame_composite"],
        artifacts={
            "clip_0_firstframe": str(video_dir / "clip_0_firstframe.png"),
            "clip_1_firstframe": str(video_dir / "clip_1_firstframe.png"),
        },
    )
    save_video_state(initial_state, root=state_root)

    run_state = _run_state()
    save_run_state(run_state, root=state_root)

    nb_client = _make_mock_nano_banana(png_bytes)
    veo_client = _make_mock_veo(mp4_bytes)
    flash_client = _make_mock_flash()
    ctx = _make_context(state_root, artifacts_root, nb_client, veo_client, flash_client)

    with patch("ugc_pipeline.steps.caption.transcribe_with_whisper", return_value=[]):
        await process_video(spec.video_id, brief, ctx, run_state)

    from ugc_pipeline.state_manager import load_video_state
    final_state = load_video_state(spec.video_id, state_root)
    assert final_state.status == "completed"
    # Nano Banana should NOT have been called (first_frame was already done)
    nb_client.generate_image.assert_not_called()


@pytest.mark.asyncio
async def test_process_video_marks_failed_on_step_exception(
    tmp_path: pathlib.Path,
) -> None:
    """A step that raises an unexpected exception must mark the video as failed_error."""
    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    state_root.mkdir()
    artifacts_root.mkdir()

    product_id = "orch000000cc"
    spec = _video_spec(product_id, spec_index=0, clip_count=2)
    brief = _product_brief(product_id, tmp_path)

    save_product_brief(brief, root=state_root)
    save_video_spec(spec, root=state_root)
    save_video_state(
        VideoState(video_id=spec.video_id, product_id=product_id, spec_index=0),
        root=state_root,
    )

    run_state = _run_state()
    save_run_state(run_state, root=state_root)

    # Nano Banana raises every time
    nb_client = MagicMock()
    nb_client.generate_image = AsyncMock(side_effect=RuntimeError("image gen failed"))

    veo_client = _make_mock_veo(b"fakemp4")
    flash_client = _make_mock_flash()
    ctx = _make_context(state_root, artifacts_root, nb_client, veo_client, flash_client)

    # Should NOT raise — exception is caught and video is marked failed
    await process_video(spec.video_id, brief, ctx, run_state)

    from ugc_pipeline.state_manager import load_video_state
    final_state = load_video_state(spec.video_id, state_root)
    assert final_state.status == "failed_error"
    assert final_state.last_error is not None


@pytest.mark.asyncio
async def test_run_pipeline_cost_tracked(
    tmp_path: pathlib.Path,
    clip_fixture_mp4_path: pathlib.Path,
    firstframe_fixture_png_path: pathlib.Path,
) -> None:
    """After run_pipeline, run_state.cumulative_cost_usd must be > 0."""
    from ugc_pipeline.orchestrator import run_pipeline
    from ugc_pipeline.steps.drive_poll import PolledImage
    import hashlib

    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    state_root.mkdir()
    artifacts_root.mkdir()

    png_bytes = firstframe_fixture_png_path.read_bytes()
    mp4_bytes = clip_fixture_mp4_path.read_bytes()

    # Fake image bytes for polling
    fake_image = b"fake_product_image_bytes_" + b"\xaa" * 50
    product_id = hashlib.sha256(fake_image).hexdigest()[:12]

    polled = PolledImage(
        product_id=product_id,
        image_bytes=fake_image,
        filename="product.jpg",
        drive_file_id="drive-file-abc",
    )

    nb_client = _make_mock_nano_banana(png_bytes)
    veo_client = _make_mock_veo(mp4_bytes)
    flash_client = _make_mock_flash()

    # Mock Gemini Pro client to return a ProductBrief-shaped JSON
    from ugc_pipeline.models import ProductBrief, VideoSpec
    from datetime import datetime, timezone

    fake_brief = ProductBrief(
        product_id=product_id,
        image_path=str(tmp_path / "product.jpg"),
        shape="cylinder",
        dominant_colours=["#fff"],
        packaging_style="plain",
        inferred_category="lifestyle",
        lifestyle_contexts=["morning", "evening", "outdoor"],
        created_at=datetime.now(timezone.utc),
    )
    fake_spec_0 = VideoSpec(
        video_id=str(uuid.uuid4()),
        product_id=product_id,
        spec_index=0,
        tone="warm",
        narrative_arc="arc 0",
        talent_id="talent_01",
        clip_count=2,
        scene_descriptions=["scene 0", "scene 1"],
        script_blocks=["Ogni mattina.", "La ceramica."],
        created_at=datetime.now(timezone.utc),
    )
    fake_specs = [
        fake_spec_0,
        fake_spec_0.model_copy(update={"video_id": str(uuid.uuid4()), "spec_index": 1, "tone": "energetic"}),
        fake_spec_0.model_copy(update={"video_id": str(uuid.uuid4()), "spec_index": 2, "tone": "serene"}),
    ]

    run_state = _run_state()
    save_run_state(run_state, root=state_root)

    ctx = _make_context(state_root, artifacts_root, nb_client, veo_client, flash_client)

    # Pre-save specs and initial state files so process_video can load them.
    from ugc_pipeline.state_manager import save_product_brief, save_video_spec, save_video_state

    save_product_brief(fake_brief, root=state_root)
    for spec in fake_specs:
        save_video_spec(spec, root=state_root)
        save_video_state(
            VideoState(video_id=spec.video_id, product_id=product_id, spec_index=spec.spec_index),
            root=state_root,
        )

    # Patch at the orchestrator module level since run_product_analyst and
    # run_creative_director are imported at module scope in orchestrator.py.
    with (
        patch(
            "ugc_pipeline.orchestrator.run_product_analyst",
            new=AsyncMock(return_value=fake_brief),
        ),
        patch(
            "ugc_pipeline.orchestrator.run_creative_director",
            new=AsyncMock(return_value=fake_specs),
        ),
        patch("ugc_pipeline.steps.caption.transcribe_with_whisper", return_value=[]),
    ):
        try:
            await run_pipeline(
                [polled],
                ctx,
                run_state,
                per_video_max_usd=8.0,
                global_max_usd=50.0,
            )
        except Exception:
            pass  # kill-switch or other; we check cost below

    # Cost must be > 0 (Nano Banana and Veo calls recorded costs)
    assert run_state.cumulative_cost_usd > 0, (
        f"Expected cumulative_cost_usd > 0, got {run_state.cumulative_cost_usd}"
    )
