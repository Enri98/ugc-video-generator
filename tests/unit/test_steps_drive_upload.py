"""Unit tests for src/ugc_pipeline/steps/drive_upload.py.

Covers:
- Skip-no-credentials path: logs WARNING, sets status="completed_local", no raise.
- Happy path with InMemoryDriveClient: uploads both files, sets drive_*_file_id.
- Idempotency: second call skips re-upload.
- Fail-fast intent from §10: DriveAuthError raised when credentials path is None.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from ugc_pipeline.models import CostBreakdown, VideoState
from ugc_pipeline.steps.drive_upload import run_drive_upload
from ugc_pipeline.utils.drive import DriveAuthError, InMemoryDriveClient, make_drive_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_video_state(tmp_path: Path, *, video_id: str | None = None) -> VideoState:
    """Return a VideoState with final + report artifacts on disk."""
    vid = video_id or str(uuid.uuid4())
    pid = "abc123def456"

    # Create fake artifact files
    video_dir = tmp_path / "artifacts" / vid
    video_dir.mkdir(parents=True, exist_ok=True)

    final_path = video_dir / "final.mp4"
    final_path.write_bytes(b"\x00" * 512)  # fake MP4

    report_path = video_dir / "report.txt"
    report_path.write_text("UGC PIPELINE REPORT\n===================\n", encoding="utf-8")

    state = VideoState(
        video_id=vid,
        product_id=pid,
        spec_index=0,
        status="completed",
        costs_usd=CostBreakdown(),
    )
    state.artifacts["final"] = str(final_path)
    state.artifacts["report"] = str(report_path)
    return state


_VIDEOS_FOLDER = "drive-videos-output"
_REPORTS_FOLDER = "drive-reports-output"


# ---------------------------------------------------------------------------
# Skip-no-credentials path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skip_when_client_is_none_sets_status(tmp_path: Path) -> None:
    state = _make_video_state(tmp_path)
    await run_drive_upload(
        state,
        client=None,
        output_videos_folder_id=_VIDEOS_FOLDER,
        output_reports_folder_id=_REPORTS_FOLDER,
    )
    assert state.status == "completed_local"


@pytest.mark.asyncio
async def test_skip_when_client_is_none_does_not_raise(tmp_path: Path) -> None:
    state = _make_video_state(tmp_path)
    # Must not raise any exception
    await run_drive_upload(
        state,
        client=None,
        output_videos_folder_id=_VIDEOS_FOLDER,
        output_reports_folder_id=_REPORTS_FOLDER,
    )


@pytest.mark.asyncio
async def test_skip_when_client_is_none_logs_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """structlog writes to stdout in test mode, so we capture stdout."""
    state = _make_video_state(tmp_path)
    await run_drive_upload(
        state,
        client=None,
        output_videos_folder_id=_VIDEOS_FOLDER,
        output_reports_folder_id=_REPORTS_FOLDER,
    )
    captured = capsys.readouterr()
    assert "drive_upload_skipped_no_credentials" in captured.out


@pytest.mark.asyncio
async def test_skip_when_video_folder_is_none(tmp_path: Path) -> None:
    state = _make_video_state(tmp_path)
    client = InMemoryDriveClient()
    await run_drive_upload(
        state,
        client=client,
        output_videos_folder_id=None,
        output_reports_folder_id=_REPORTS_FOLDER,
    )
    assert state.status == "completed_local"


@pytest.mark.asyncio
async def test_skip_when_reports_folder_is_none(tmp_path: Path) -> None:
    state = _make_video_state(tmp_path)
    client = InMemoryDriveClient()
    await run_drive_upload(
        state,
        client=client,
        output_videos_folder_id=_VIDEOS_FOLDER,
        output_reports_folder_id=None,
    )
    assert state.status == "completed_local"


@pytest.mark.asyncio
async def test_skip_does_not_set_drive_file_ids(tmp_path: Path) -> None:
    state = _make_video_state(tmp_path)
    await run_drive_upload(
        state,
        client=None,
        output_videos_folder_id=_VIDEOS_FOLDER,
        output_reports_folder_id=_REPORTS_FOLDER,
    )
    assert state.drive_video_file_id is None
    assert state.drive_report_file_id is None


# ---------------------------------------------------------------------------
# Happy path with InMemoryDriveClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_sets_drive_video_file_id(tmp_path: Path) -> None:
    state = _make_video_state(tmp_path)
    client = InMemoryDriveClient()
    await run_drive_upload(
        state,
        client=client,
        output_videos_folder_id=_VIDEOS_FOLDER,
        output_reports_folder_id=_REPORTS_FOLDER,
    )
    assert state.drive_video_file_id is not None
    assert state.drive_video_file_id.startswith("mem-")


@pytest.mark.asyncio
async def test_happy_path_sets_drive_report_file_id(tmp_path: Path) -> None:
    state = _make_video_state(tmp_path)
    client = InMemoryDriveClient()
    await run_drive_upload(
        state,
        client=client,
        output_videos_folder_id=_VIDEOS_FOLDER,
        output_reports_folder_id=_REPORTS_FOLDER,
    )
    assert state.drive_report_file_id is not None
    assert state.drive_report_file_id.startswith("mem-")


@pytest.mark.asyncio
async def test_happy_path_video_appears_in_drive(tmp_path: Path) -> None:
    state = _make_video_state(tmp_path)
    client = InMemoryDriveClient()
    await run_drive_upload(
        state,
        client=client,
        output_videos_folder_id=_VIDEOS_FOLDER,
        output_reports_folder_id=_REPORTS_FOLDER,
    )
    files = client.list_files(_VIDEOS_FOLDER)
    assert len(files) == 1
    assert files[0].mime_type == "video/mp4"


@pytest.mark.asyncio
async def test_happy_path_report_appears_in_drive(tmp_path: Path) -> None:
    state = _make_video_state(tmp_path)
    client = InMemoryDriveClient()
    await run_drive_upload(
        state,
        client=client,
        output_videos_folder_id=_VIDEOS_FOLDER,
        output_reports_folder_id=_REPORTS_FOLDER,
    )
    files = client.list_files(_REPORTS_FOLDER)
    assert len(files) == 1
    assert files[0].mime_type == "text/plain"


@pytest.mark.asyncio
async def test_happy_path_uploaded_content_matches(tmp_path: Path) -> None:
    state = _make_video_state(tmp_path)
    client = InMemoryDriveClient()
    await run_drive_upload(
        state,
        client=client,
        output_videos_folder_id=_VIDEOS_FOLDER,
        output_reports_folder_id=_REPORTS_FOLDER,
    )
    # Check video bytes
    vid_fid = state.drive_video_file_id
    assert vid_fid is not None
    uploaded_bytes = client.download_file(vid_fid)
    assert uploaded_bytes == Path(state.artifacts["final"]).read_bytes()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotency_skips_second_upload(tmp_path: Path) -> None:
    state = _make_video_state(tmp_path)
    client = InMemoryDriveClient()

    await run_drive_upload(
        state,
        client=client,
        output_videos_folder_id=_VIDEOS_FOLDER,
        output_reports_folder_id=_REPORTS_FOLDER,
    )
    first_video_fid = state.drive_video_file_id
    first_report_fid = state.drive_report_file_id

    # Second call must be a no-op
    await run_drive_upload(
        state,
        client=client,
        output_videos_folder_id=_VIDEOS_FOLDER,
        output_reports_folder_id=_REPORTS_FOLDER,
    )
    assert state.drive_video_file_id == first_video_fid
    assert state.drive_report_file_id == first_report_fid

    # Only 1 file per folder after idempotent call
    assert len(client.list_files(_VIDEOS_FOLDER)) == 1
    assert len(client.list_files(_REPORTS_FOLDER)) == 1


# ---------------------------------------------------------------------------
# §10 Fail-fast intent: DriveAuthError when credentials not configured
# ---------------------------------------------------------------------------


def test_make_drive_client_raises_on_none() -> None:
    """Explicit §10 fail-fast test: pipeline must not start without credentials."""
    with pytest.raises(DriveAuthError):
        make_drive_client(None)


def test_make_drive_client_raises_on_missing_file(tmp_path: Path) -> None:
    """Fail-fast when credentials file doesn't exist."""
    fake_path = tmp_path / "missing" / "service-account.json"
    with pytest.raises(DriveAuthError) as exc_info:
        make_drive_client(fake_path)
    assert "Drive integration is required" in str(exc_info.value)
