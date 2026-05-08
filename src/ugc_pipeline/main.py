"""Pipeline entry point for the UGC video generator.

Implements SPEC.md §10 (one-shot scan-and-exit execution mode).

Usage:
    python -m ugc_pipeline.main [--dry-run] [--mock-everything]
    python -m ugc_pipeline               # via __main__.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys
import uuid
from datetime import datetime, timezone


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ugc_pipeline.main",
        description="UGC video pipeline — one-shot scan-and-exit (SPEC.md §10).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run text/vision steps for real; skip paid Veo and Nano Banana calls.",
    )
    parser.add_argument(
        "--mock-everything",
        action="store_true",
        default=False,
        help=(
            "Wire in InMemoryDriveClient and no-op Veo/Nano Banana clients. "
            "No paid API calls; no Drive credentials required. "
            "Intended for local smoke-testing."
        ),
    )
    return parser


def cli_entry(argv: list[str] | None = None) -> None:
    """Top-level pipeline entry point.

    Loads .env, configures logging, builds all clients, polls Drive, and runs
    the async pipeline. Exits with code 0 on success, 1 on unhandled error,
    2 if the kill-switch fires.
    """
    # Parse CLI args before importing anything that might fail (faster feedback).
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    dry_run: bool = args.dry_run
    mock_everything: bool = args.mock_everything

    # Load .env (ignore if absent — env vars may already be set)
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed; proceed with env as-is

    # Configure structlog + stdlib logging
    from ugc_pipeline.utils.logging import configure_logging
    from ugc_pipeline.utils.config import load_pipeline_config, load_talent_pool

    cfg = load_pipeline_config()
    log_cfg = cfg.get("logging", {})
    configure_logging(
        log_file=log_cfg.get("log_file", "logs/pipeline.log"),
        level=os.environ.get("LOG_LEVEL", log_cfg.get("level", "INFO")),
        max_bytes=int(log_cfg.get("max_bytes", 10_485_760)),
        backup_count=int(log_cfg.get("backup_count", 5)),
    )

    import structlog
    log = structlog.get_logger(__name__)

    # Load talent pool
    try:
        talent_pool = load_talent_pool()
    except FileNotFoundError as exc:
        log.error("talent_pool_not_found", error=str(exc))
        sys.exit(1)

    # Build clients
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    creds_path_str = os.environ.get("GOOGLE_DRIVE_CREDENTIALS_PATH", "")
    input_folder_id = os.environ.get("GOOGLE_DRIVE_INPUT_FOLDER_ID", "")
    output_videos_folder_id = os.environ.get("GOOGLE_DRIVE_OUTPUT_VIDEOS_FOLDER_ID", "")
    output_reports_folder_id = os.environ.get("GOOGLE_DRIVE_OUTPUT_REPORTS_FOLDER_ID", "")

    # Drive client
    if mock_everything:
        from ugc_pipeline.utils.drive import InMemoryDriveClient
        drive_client = InMemoryDriveClient()
        log.info("mock_drive_client_active")
    else:
        from ugc_pipeline.utils.drive import DriveAuthError, make_drive_client
        creds_path = pathlib.Path(creds_path_str) if creds_path_str else None
        try:
            drive_client = make_drive_client(creds_path)
        except DriveAuthError as exc:
            print(
                f"ERROR: Drive credentials not found at {creds_path_str!r}.\n"
                "Drive integration is required for input polling and output upload.\n"
                "See §16 Day 6 in SPEC.md for the service account bootstrap procedure.\n"
                "Set GOOGLE_DRIVE_CREDENTIALS_PATH in .env once credentials are in place.",
                file=sys.stderr,
            )
            log.critical("drive_auth_error", error=str(exc))
            sys.exit(1)

    # Gemini Pro client (text + vision)
    if mock_everything:
        from unittest.mock import AsyncMock, MagicMock
        gemini_pro_client = MagicMock()
        gemini_pro_client.generate_content = AsyncMock(return_value=MagicMock(text="{}", usage_metadata=None))
    else:
        from ugc_pipeline.steps.product_analyst import make_default_client as _make_gemini
        gemini_pro_client = _make_gemini(google_api_key)

    # Nano Banana client
    if mock_everything or dry_run:
        from unittest.mock import AsyncMock, MagicMock
        from ugc_pipeline.steps.first_frame import NanoBananaResult
        nano_banana_client = MagicMock()
        nano_banana_client.generate_image = AsyncMock(
            return_value=NanoBananaResult(png_bytes=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        )
    else:
        from ugc_pipeline.steps.first_frame import make_default_client as _make_nb
        nano_banana_client = _make_nb(google_api_key)

    # Veo client
    if mock_everything or dry_run:
        from unittest.mock import AsyncMock, MagicMock
        veo_client = MagicMock()
        veo_client.submit = AsyncMock(return_value="mock-op-001")
        veo_client.poll = AsyncMock(
            return_value={"done": True, "mp4_bytes": b"fakemp4", "error": None, "safety_block": False}
        )
    else:
        # Real Veo client — Day 8 wiring; for now raise informative error.
        log.error("veo_client_not_wired", reason="Real Veo client requires Day 8 setup.")
        raise NotImplementedError(
            "Real Veo client is not yet wired. Use --mock-everything for smoke testing."
        )

    # Gemini Flash client (safety retry)
    if mock_everything:
        from unittest.mock import AsyncMock, MagicMock
        flash_client = MagicMock()
        flash_client.rewrite = AsyncMock(return_value="Rewritten safe scene prompt.")
    else:
        # Share the same google-genai client; wrap in a thin adapter
        class _FlashAdapter:
            def __init__(self, models_client: object) -> None:
                self._c = models_client

            async def rewrite(self, prompt: str) -> str:
                resp = await self._c.generate_content(  # type: ignore[union-attr]
                    model="gemini-2.0-flash",
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                )
                return resp.text or prompt

        flash_client = _FlashAdapter(gemini_pro_client)

    # Rate limiters (aiolimiter)
    try:
        from aiolimiter import AsyncLimiter  # type: ignore[import-untyped]
        gemini_limiter = AsyncLimiter(60, 60)   # 60 RPM
        nano_banana_limiter = AsyncLimiter(30, 60)  # 30 RPM
    except ImportError:
        gemini_limiter = None
        nano_banana_limiter = None

    # Veo semaphore
    parallelism_cfg = cfg.get("parallelism", {})
    veo_semaphore = asyncio.Semaphore(int(parallelism_cfg.get("max_concurrent_veo_ops", 4)))

    # State + artifact directories
    paths_cfg = cfg.get("paths", {})
    state_root = pathlib.Path(paths_cfg.get("state_dir", "state"))
    artifacts_root = pathlib.Path(paths_cfg.get("artifacts_dir", "artifacts"))
    state_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    # Orchestrator context
    from ugc_pipeline.orchestrator import OrchestratorContext
    ctx = OrchestratorContext(
        gemini_pro_client=gemini_pro_client,
        nano_banana_client=nano_banana_client,
        veo_client=veo_client,
        flash_client=flash_client,
        drive_client=drive_client,
        state_root=state_root,
        artifacts_root=artifacts_root,
        output_videos_folder_id=output_videos_folder_id or None,
        output_reports_folder_id=output_reports_folder_id or None,
        cfg=cfg,
        talent_pool=talent_pool,
        dry_run=dry_run,
        gemini_limiter=gemini_limiter,
        nano_banana_limiter=nano_banana_limiter,
        veo_semaphore=veo_semaphore,
    )

    # Build RunState
    from ugc_pipeline.models import RunState
    from ugc_pipeline.state_manager import save_run_state
    run_id = str(uuid.uuid4())
    run_state = RunState(run_id=run_id)
    save_run_state(run_state, root=state_root)
    log.info("pipeline_started", run_id=run_id, dry_run=dry_run, mock_everything=mock_everything)

    # Poll Drive for new product images
    if mock_everything:
        images: list = []
        log.info("mock_drive_poll_active", message="No images polled in --mock-everything mode.")
    else:
        from ugc_pipeline.steps.drive_poll import poll_drive_input
        if not input_folder_id:
            log.error("drive_input_folder_id_missing")
            print("ERROR: GOOGLE_DRIVE_INPUT_FOLDER_ID is not set in .env", file=sys.stderr)
            sys.exit(1)
        try:
            images = poll_drive_input(drive_client, input_folder_id, state_root=state_root)
        except Exception as exc:
            log.critical("drive_poll_failed", error=str(exc))
            sys.exit(1)

    log.info("drive_poll_complete", new_images=len(images))

    # Run the pipeline
    budget_cfg = cfg.get("budget", {})
    per_video_max_usd = float(budget_cfg.get("per_video_max_usd", 8.0))
    global_max_usd = float(budget_cfg.get("global_max_usd", 50.0))

    from ugc_pipeline.cost_tracker import KillSwitchFiredError
    from ugc_pipeline.orchestrator import run_pipeline

    try:
        asyncio.run(
            run_pipeline(
                images,
                ctx,
                run_state,
                per_video_max_usd=per_video_max_usd,
                global_max_usd=global_max_usd,
            )
        )
    except KillSwitchFiredError:
        print(
            f"\nKILL SWITCH FIRED: Global budget cap of ${global_max_usd:.2f} reached.\n"
            f"Cumulative cost this run: ${run_state.cumulative_cost_usd:.2f}\n"
            "RunState saved. In-flight Veo operations drained.\n"
            "To continue processing remaining products, increase budget.global_max_usd\n"
            f"in pipeline_config.yaml and run: python -m ugc_pipeline.main",
            file=sys.stderr,
        )
        sys.exit(2)
    except Exception as exc:
        log.critical("pipeline_unhandled_exception", error=str(exc))
        sys.exit(1)

    log.info(
        "pipeline_exit_ok",
        run_id=run_id,
        videos_completed=run_state.videos_completed,
        videos_failed=run_state.videos_failed,
        cumulative_cost_usd=run_state.cumulative_cost_usd,
    )
    sys.exit(0)


if __name__ == "__main__":
    cli_entry()
