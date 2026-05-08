"""Drive upload step for the UGC pipeline.

Implements SPEC.md §5 Step 11.

Uploads the final captioned MP4 and the TXT report to their respective Drive
output folders. Records the Drive file IDs in VideoState.

Idempotent: if both drive_video_file_id and drive_report_file_id are already
set, the step is skipped.

Skip-with-warning path: if Drive credentials are not configured (client is
None) or the output folder IDs are not set, the step emits a WARNING and
marks the video as ``completed_local`` rather than raising.

The underlying Drive SDK calls are synchronous; this step wraps them with
``asyncio.to_thread`` to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from ugc_pipeline.models import VideoState
from ugc_pipeline.utils.drive import DriveAuthError, DriveClientProtocol

log = structlog.get_logger(__name__)

_SKIP_MESSAGE = (
    "Drive credentials are not configured. "
    "The final MP4 and report have been saved locally but not uploaded. "
    "See §16 Day 6 in SPEC.md for the service account bootstrap procedure. "
    "Set GOOGLE_DRIVE_CREDENTIALS_PATH in .env to enable upload."
)


async def run_drive_upload(
    video_state: VideoState,
    *,
    client: DriveClientProtocol | None,
    output_videos_folder_id: str | None,
    output_reports_folder_id: str | None,
) -> None:
    """Upload MP4 and TXT report to Drive output folders.

    Parameters
    ----------
    video_state:
        Mutable VideoState. ``drive_video_file_id`` and
        ``drive_report_file_id`` are set on successful upload.
        ``status`` is set to ``"completed_local"`` when the upload is
        skipped due to missing credentials.
    client:
        A ``DriveClientProtocol`` implementation, or ``None`` when Drive
        is not configured.
    output_videos_folder_id:
        Drive folder ID for MP4 uploads, or ``None``.
    output_reports_folder_id:
        Drive folder ID for TXT report uploads, or ``None``.

    Raises
    ------
    DriveAuthError
        Re-raised if the underlying client throws an authentication error
        (terminal — local artifacts are preserved for manual upload).
    KeyError
        If ``final`` or ``report`` artifact keys are missing from
        *video_state* and the upload is attempted.
    """
    # ------------------------------------------------------------------
    # Idempotency check
    # ------------------------------------------------------------------
    if (
        video_state.drive_video_file_id is not None
        and video_state.drive_report_file_id is not None
    ):
        log.info(
            "step_skipped_idempotent",
            step="drive_upload",
            video_id=video_state.video_id,
        )
        return

    # ------------------------------------------------------------------
    # Skip-with-warning if credentials / folder IDs are absent
    # ------------------------------------------------------------------
    if (
        client is None
        or output_videos_folder_id is None
        or output_reports_folder_id is None
    ):
        log.warning(
            "drive_upload_skipped_no_credentials",
            video_id=video_state.video_id,
            message=_SKIP_MESSAGE,
        )
        video_state.status = "completed_local"
        return

    log.info(
        "step_started",
        step="drive_upload",
        video_id=video_state.video_id,
    )

    # ------------------------------------------------------------------
    # Upload final MP4
    # ------------------------------------------------------------------
    final_path = Path(video_state.artifacts["final"])
    final_name = f"{video_state.video_id}_final.mp4"
    try:
        video_file_id = await asyncio.to_thread(
            client.upload_file,
            folder_id=output_videos_folder_id,
            name=final_name,
            mime_type="video/mp4",
            content=final_path.read_bytes(),
        )
    except DriveAuthError:
        log.error(
            "drive_upload_auth_error",
            video_id=video_state.video_id,
            file=final_name,
        )
        raise

    video_state.drive_video_file_id = video_file_id
    log.info(
        "drive_upload_video_ok",
        video_id=video_state.video_id,
        drive_file_id=video_file_id,
    )

    # ------------------------------------------------------------------
    # Upload TXT report
    # ------------------------------------------------------------------
    report_path = Path(video_state.artifacts["report"])
    report_name = f"{video_state.video_id}_report.txt"
    try:
        report_file_id = await asyncio.to_thread(
            client.upload_file,
            folder_id=output_reports_folder_id,
            name=report_name,
            mime_type="text/plain",
            content=report_path.read_bytes(),
        )
    except DriveAuthError:
        log.error(
            "drive_upload_auth_error",
            video_id=video_state.video_id,
            file=report_name,
        )
        raise

    video_state.drive_report_file_id = report_file_id
    log.info(
        "drive_upload_report_ok",
        video_id=video_state.video_id,
        drive_file_id=report_file_id,
    )

    log.info(
        "step_completed",
        step="drive_upload",
        video_id=video_state.video_id,
        video_drive_file_id=video_file_id,
        report_drive_file_id=report_file_id,
    )
