"""Operational CLI for the UGC pipeline.

Subcommands:
  list-failed       — print all videos in any failed_* state.
  retry <video_id>  — reset a failed video to pending.
  reset <video_id>  — delete artifacts and reinitialise VideoState.
  status            — print the most recent RunState summary.
  dry-run-cost      — estimate downstream costs for one product.

See SPEC.md §11 for the full operational recovery contract.

Usage:
    python -m ugc_pipeline.cli <subcommand> [args]
    cli_main([...])           # programmatic entry point for tests
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_root() -> pathlib.Path:
    """Return the state root directory (relative to CWD)."""
    return pathlib.Path("state")


def _artifacts_root() -> pathlib.Path:
    return pathlib.Path("artifacts")


def _load_video_state(video_id: str, state_root: pathlib.Path) -> dict[str, Any] | None:
    """Load raw VideoState dict, or None if the file does not exist."""
    path = state_root / "videos" / f"{video_id}.state.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_video_state_raw(video_id: str, data: dict[str, Any], state_root: pathlib.Path) -> None:
    """Write a raw VideoState dict atomically."""
    import os
    import tempfile

    path = state_root / "videos" / f"{video_id}.state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Subcommand: list-failed
# ---------------------------------------------------------------------------


def cmd_list_failed(args: argparse.Namespace) -> int:
    """Print all videos in any failed_* state."""
    state_root = pathlib.Path(args.state_dir) if hasattr(args, "state_dir") and args.state_dir else _state_root()
    videos_dir = state_root / "videos"

    if not videos_dir.is_dir():
        print("No state directory found — no failed videos.")
        return 0

    found = False
    for state_file in sorted(videos_dir.glob("*.state.json")):
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = data.get("status", "")
        if not status.startswith("failed"):
            continue
        found = True
        print(
            f"video_id={data.get('video_id', '?')!s}  "
            f"status={status}  "
            f"current_step={data.get('current_step') or '-'}  "
            f"last_error={data.get('last_error') or '-'}"
        )

    if not found:
        print("No failed videos found.")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: retry
# ---------------------------------------------------------------------------


def cmd_retry(args: argparse.Namespace) -> int:
    """Reset a failed video to pending so it can be retried."""
    state_root = pathlib.Path(args.state_dir) if hasattr(args, "state_dir") and args.state_dir else _state_root()
    video_id: str = args.video_id

    data = _load_video_state(video_id, state_root)
    if data is None:
        print(f"ERROR: No state file found for video_id={video_id!r}", file=sys.stderr)
        return 1

    status = data.get("status", "")
    if not status.startswith("failed"):
        print(f"Video {video_id!r} is not in a failed state (current: {status!r}). Nothing to retry.")
        return 0

    data["status"] = "pending"
    data["last_error"] = None
    _save_video_state_raw(video_id, data, state_root)
    print(f"Video {video_id!r} reset to pending. Run the pipeline again to retry.")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: reset
# ---------------------------------------------------------------------------


def cmd_reset(args: argparse.Namespace) -> int:
    """Delete all artifacts for a video and reinitialise its VideoState."""
    state_root = pathlib.Path(args.state_dir) if hasattr(args, "state_dir") and args.state_dir else _state_root()
    artifacts_root = _artifacts_root()
    video_id: str = args.video_id

    # Delete local artifacts directory
    artifact_dir = artifacts_root / video_id
    if artifact_dir.is_dir():
        shutil.rmtree(artifact_dir)
        print(f"Deleted artifact directory: {artifact_dir}")

    # Load existing state to preserve product_id and spec_index
    existing = _load_video_state(video_id, state_root)
    if existing is None:
        print(f"No state file found for video_id={video_id!r}. Nothing to reset.")
        return 0

    # Reinitialise state (keep .spec.json — operator can re-derive)
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    fresh_state = {
        "video_id": video_id,
        "product_id": existing.get("product_id", ""),
        "spec_index": existing.get("spec_index", 0),
        "status": "pending",
        "current_step": None,
        "completed_steps": [],
        "artifacts": {},
        "costs_usd": {
            "product_analyst_usd": 0.0,
            "creative_director_usd": 0.0,
            "first_frame_usd": 0.0,
            "veo_usd": 0.0,
            "safety_retry_usd": 0.0,
            "caption_usd": 0.0,
        },
        "prompt_versions": {},
        "last_error": None,
        "drive_video_file_id": None,
        "drive_report_file_id": None,
        "created_at": now,
        "updated_at": now,
    }
    _save_video_state_raw(video_id, fresh_state, state_root)
    print(f"Video {video_id!r} state reset to pending with empty artifacts.")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    """Print the most recent RunState summary."""
    state_root = pathlib.Path(args.state_dir) if hasattr(args, "state_dir") and args.state_dir else _state_root()
    runs_dir = state_root / "runs"

    if not runs_dir.is_dir():
        print("No run state found.")
        return 0

    run_files = sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_files:
        print("No run state found.")
        return 0

    run_file = run_files[0]
    try:
        run_data = json.loads(run_file.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: Could not read run state: {exc}", file=sys.stderr)
        return 1

    print("=== Run State ===")
    print(f"run_id:               {run_data.get('run_id', '?')}")
    print(f"started_at:           {run_data.get('started_at', '?')}")
    print(f"ended_at:             {run_data.get('ended_at') or 'still running'}")
    print(f"cumulative_cost_usd:  ${run_data.get('cumulative_cost_usd', 0.0):.4f}")
    print(f"kill_switch_fired:    {run_data.get('kill_switch_fired', False)}")
    print(f"videos_completed:     {run_data.get('videos_completed', 0)}")
    print(f"videos_failed:        {run_data.get('videos_failed', 0)}")

    products_seen: list[str] = run_data.get("products_seen", [])
    if products_seen:
        print("\n=== Products ===")
        for product_id in products_seen:
            print(f"\nProduct: {product_id}")
            videos_dir = state_root / "videos"
            if videos_dir.is_dir():
                for state_file in sorted(videos_dir.glob("*.state.json")):
                    try:
                        vs = json.loads(state_file.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if vs.get("product_id") != product_id:
                        continue
                    print(
                        f"  video_id={vs.get('video_id', '?')!s}  "
                        f"spec_index={vs.get('spec_index', '?')}  "
                        f"status={vs.get('status', '?')}"
                    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: dry-run-cost
# ---------------------------------------------------------------------------


def cmd_dry_run_cost(args: argparse.Namespace) -> int:
    """Estimate downstream costs for one product (text/vision steps only)."""
    from ugc_pipeline.steps.first_frame import estimate_first_frame_cost_usd
    from ugc_pipeline.steps.veo import estimate_veo_clip_cost_usd
    from ugc_pipeline.steps.product_analyst import estimate_product_analyst_cost_usd
    from ugc_pipeline.steps.creative_director import estimate_creative_director_cost_usd

    product_id: str = args.product_id
    mock_gemini: bool = getattr(args, "mock_gemini", False)

    # Flat cost estimates (not requiring real API calls)
    analyst_est = 0.02  # midpoint estimate per SPEC.md §12
    director_est_per_spec = 0.01  # midpoint estimate
    first_frame_est_per_clip = estimate_first_frame_cost_usd()
    veo_est_per_clip = estimate_veo_clip_cost_usd()

    # Assume 3 specs × 2 clips each (conservative)
    specs = 3
    clips_per_spec = 2

    director_total = director_est_per_spec * specs
    first_frame_total = first_frame_est_per_clip * specs * clips_per_spec
    veo_total = veo_est_per_clip * specs * clips_per_spec

    grand_total = analyst_est + director_total + first_frame_total + veo_total

    print(f"=== Dry-run cost estimate for product {product_id!r} ===")
    print(f"(Assumes {specs} specs × {clips_per_spec} clips each)")
    print()
    print(f"  product_analyst (Gemini 2.5 Pro vision):  ${analyst_est:.4f}")
    print(f"  creative_director ({specs} specs):             ${director_total:.4f}")
    print(f"  first_frame_composite ({specs * clips_per_spec} clips):      ${first_frame_total:.4f}")
    print(f"  veo_generation ({specs * clips_per_spec} clips):             ${veo_total:.4f}")
    print(f"  ----------------------------------------")
    print(f"  ESTIMATED TOTAL:                          ${grand_total:.4f}")
    print()
    print("Note: actual costs depend on token counts, clip durations, and API tier.")
    if mock_gemini:
        print("(--mock-gemini flag set: text/vision calls were not made)")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ugc_pipeline.cli",
        description="Operational CLI for the UGC video pipeline (SPEC.md §11).",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Override the default state directory (default: state/).",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # list-failed
    subparsers.add_parser("list-failed", help="Print all videos in a failed_* state.")

    # retry
    retry_p = subparsers.add_parser("retry", help="Reset a failed video to pending.")
    retry_p.add_argument("video_id", help="UUID of the video to retry.")

    # reset
    reset_p = subparsers.add_parser("reset", help="Delete artifacts and reinitialise VideoState.")
    reset_p.add_argument("video_id", help="UUID of the video to reset.")

    # status
    subparsers.add_parser("status", help="Print the most recent RunState summary.")

    # dry-run-cost
    drc_p = subparsers.add_parser(
        "dry-run-cost",
        help="Estimate downstream costs for one product.",
    )
    drc_p.add_argument("product_id", help="Product ID to estimate costs for.")
    drc_p.add_argument(
        "--mock-gemini",
        action="store_true",
        default=False,
        help="Skip real Gemini calls; use static estimates only.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

_SUBCOMMAND_HANDLERS = {
    "list-failed": cmd_list_failed,
    "retry": cmd_retry,
    "reset": cmd_reset,
    "status": cmd_status,
    "dry-run-cost": cmd_dry_run_cost,
}


def cli_main(argv: list[str] | None = None) -> int:
    """Parse *argv* and dispatch to the appropriate subcommand handler.

    Returns an exit code (0 = success, non-zero = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = _SUBCOMMAND_HANDLERS.get(args.subcommand)
    if handler is None:
        print(f"Unknown subcommand: {args.subcommand!r}", file=sys.stderr)
        return 2

    return handler(args)


if __name__ == "__main__":
    sys.exit(cli_main())
