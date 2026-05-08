"""Unit tests for ugc_pipeline.cli (SPEC.md §11).

Covers:
  - Argument parsing for each subcommand.
  - list-failed output for seeded failed videos.
  - retry flips video status from failed_error to pending.
  - reset deletes artifacts directory and reinitialises VideoState.
  - status reads the most recent RunState file.
  - dry-run-cost with --mock-gemini prints a cost breakdown.
"""

from __future__ import annotations

import json
import pathlib
import uuid
from datetime import datetime, timezone

import pytest

from ugc_pipeline.cli import cli_main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_state(state_dir: pathlib.Path, video_id: str, status: str, **extra: object) -> None:
    """Write a minimal VideoState JSON for *video_id* into *state_dir/videos/*."""
    videos_dir = state_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "video_id": video_id,
        "product_id": "abc123def456",
        "spec_index": 0,
        "status": status,
        "current_step": extra.get("current_step"),
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
        "last_error": extra.get("last_error"),
        "drive_video_file_id": None,
        "drive_report_file_id": None,
        "created_at": now,
        "updated_at": now,
    }
    (videos_dir / f"{video_id}.state.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def _write_run_state(state_dir: pathlib.Path, run_id: str, **kw: object) -> None:
    runs_dir = state_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "run_id": run_id,
        "started_at": kw.get("started_at", now),
        "ended_at": None,
        "cumulative_cost_usd": float(kw.get("cumulative_cost_usd", 1.23)),
        "kill_switch_fired": bool(kw.get("kill_switch_fired", False)),
        "products_seen": list(kw.get("products_seen", ["abc123def456"])),
        "videos_completed": int(kw.get("videos_completed", 2)),
        "videos_failed": int(kw.get("videos_failed", 0)),
    }
    (runs_dir / f"{run_id}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Test: list-failed
# ---------------------------------------------------------------------------


def test_list_failed_shows_failed_videos(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    """list-failed must print video_id, status, current_step, last_error for failed videos."""
    state_dir = tmp_path / "state"
    vid1 = str(uuid.uuid4())
    vid2 = str(uuid.uuid4())

    _write_state(state_dir, vid1, "failed_error", current_step="veo_generation", last_error="network error")
    _write_state(state_dir, vid2, "completed")  # should NOT appear

    rc = cli_main(["--state-dir", str(state_dir), "list-failed"])

    assert rc == 0
    captured = capsys.readouterr().out
    assert vid1 in captured
    assert "failed_error" in captured
    assert "veo_generation" in captured
    assert "network error" in captured
    assert vid2 not in captured


def test_list_failed_no_failed_videos(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    """list-failed with no failed videos must print an appropriate message."""
    state_dir = tmp_path / "state"
    vid = str(uuid.uuid4())
    _write_state(state_dir, vid, "completed")

    rc = cli_main(["--state-dir", str(state_dir), "list-failed"])
    assert rc == 0
    assert "No failed" in capsys.readouterr().out


def test_list_failed_empty_state_dir(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    """list-failed with no state directory must print graceful message and exit 0."""
    empty_dir = tmp_path / "nonexistent_state"
    rc = cli_main(["--state-dir", str(empty_dir), "list-failed"])
    assert rc == 0


# ---------------------------------------------------------------------------
# Test: retry
# ---------------------------------------------------------------------------


def test_retry_flips_status_to_pending(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    """retry must change video status from failed_* to pending."""
    state_dir = tmp_path / "state"
    vid = str(uuid.uuid4())
    _write_state(state_dir, vid, "failed_error", last_error="some error")

    rc = cli_main(["--state-dir", str(state_dir), "retry", vid])
    assert rc == 0

    state_file = state_dir / "videos" / f"{vid}.state.json"
    data = json.loads(state_file.read_text())
    assert data["status"] == "pending"
    assert data["last_error"] is None


def test_retry_nonexistent_video(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    """retry with a non-existent video_id must exit 1."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    rc = cli_main(["--state-dir", str(state_dir), "retry", "fake-video-id"])
    assert rc == 1


def test_retry_already_pending(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    """retry on a non-failed video must print a message and return 0."""
    state_dir = tmp_path / "state"
    vid = str(uuid.uuid4())
    _write_state(state_dir, vid, "completed")

    rc = cli_main(["--state-dir", str(state_dir), "retry", vid])
    assert rc == 0


# ---------------------------------------------------------------------------
# Test: reset
# ---------------------------------------------------------------------------


def test_reset_deletes_artifacts_and_reinits(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    """reset must delete the artifact directory and reinitialise VideoState to pending."""
    state_dir = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"  # CLI uses default "artifacts/" relative to CWD

    vid = str(uuid.uuid4())
    _write_state(state_dir, vid, "failed_error", last_error="ffmpeg crash")

    # Create a fake artifact directory
    artifact_dir = artifacts_root / vid
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "final.mp4").write_bytes(b"fakevideo")

    # Override the artifacts root by temporarily patching
    import ugc_pipeline.cli as cli_mod
    original_fn = cli_mod._artifacts_root

    try:
        cli_mod._artifacts_root = lambda: artifacts_root  # type: ignore[method-assign]
        rc = cli_main(["--state-dir", str(state_dir), "reset", vid])
    finally:
        cli_mod._artifacts_root = original_fn  # type: ignore[method-assign]

    assert rc == 0
    assert not artifact_dir.exists(), "Artifact directory should have been deleted"

    state_file = state_dir / "videos" / f"{vid}.state.json"
    data = json.loads(state_file.read_text())
    assert data["status"] == "pending"
    assert data["completed_steps"] == []
    assert data["artifacts"] == {}


# ---------------------------------------------------------------------------
# Test: status
# ---------------------------------------------------------------------------


def test_status_reads_most_recent_run(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    """status must print run_id, cost, videos_completed, videos_failed."""
    state_dir = tmp_path / "state"
    run_id = str(uuid.uuid4())
    _write_run_state(
        state_dir,
        run_id,
        cumulative_cost_usd=3.45,
        videos_completed=3,
        videos_failed=1,
    )

    rc = cli_main(["--state-dir", str(state_dir), "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert run_id in out
    assert "3.4500" in out or "3.45" in out
    assert "videos_completed" in out
    assert "videos_failed" in out


def test_status_empty_state(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    """status with no run state files must print graceful message and exit 0."""
    empty = tmp_path / "empty_state"
    rc = cli_main(["--state-dir", str(empty), "status"])
    assert rc == 0
    assert "No run state" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Test: dry-run-cost
# ---------------------------------------------------------------------------


def test_dry_run_cost_prints_breakdown(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    """dry-run-cost must print a cost breakdown table."""
    rc = cli_main([
        "--state-dir", str(tmp_path / "state"),
        "dry-run-cost", "abc123def456",
        "--mock-gemini",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ESTIMATED TOTAL" in out or "estimated" in out.lower()
    assert "abc123def456" in out


def test_dry_run_cost_includes_all_stages(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    """dry-run-cost output must mention all major cost-bearing stages."""
    rc = cli_main([
        "--state-dir", str(tmp_path / "state"),
        "dry-run-cost", "abc123def456",
        "--mock-gemini",
    ])
    out = capsys.readouterr().out
    assert "product_analyst" in out or "analyst" in out.lower()
    assert "creative_director" in out or "director" in out.lower()
    assert "first_frame" in out or "nano" in out.lower()
    assert "veo" in out.lower()


# ---------------------------------------------------------------------------
# Test: argument parsing edge cases
# ---------------------------------------------------------------------------


def test_parser_requires_subcommand() -> None:
    """Calling cli_main with no subcommand must fail (exit code != 0)."""
    with pytest.raises(SystemExit) as exc_info:
        cli_main([])
    assert exc_info.value.code != 0


def test_retry_requires_video_id() -> None:
    """retry with no video_id argument must fail."""
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["retry"])
    assert exc_info.value.code != 0


def test_reset_requires_video_id() -> None:
    """reset with no video_id argument must fail."""
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["reset"])
    assert exc_info.value.code != 0
