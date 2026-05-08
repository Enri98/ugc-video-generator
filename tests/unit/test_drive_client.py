"""Unit tests for src/ugc_pipeline/utils/drive.py.

Covers:
- InMemoryDriveClient list / download / upload
- make_drive_client(None) raises DriveAuthError with exact SPEC.md §10 message
- make_drive_client with non-existent path raises DriveAuthError
"""

from __future__ import annotations

import pytest
from pathlib import Path

from ugc_pipeline.utils.drive import (
    DriveAuthError,
    DriveClientProtocol,
    DriveFile,
    DriveNotFoundError,
    InMemoryDriveClient,
    make_drive_client,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FOLDER_A = "folder-aaa"
_FOLDER_B = "folder-bbb"


@pytest.fixture()
def populated_client() -> InMemoryDriveClient:
    """InMemoryDriveClient seeded with two folders and three files."""
    return InMemoryDriveClient(
        initial={
            _FOLDER_A: [
                ("product.jpg", "image/jpeg", b"\xff\xd8\xff" + b"\x00" * 100),
                ("logo.png", "image/png", b"\x89PNG" + b"\x00" * 50),
            ],
            _FOLDER_B: [
                ("document.pdf", "application/pdf", b"%PDF" + b"\x00" * 30),
            ],
        }
    )


# ---------------------------------------------------------------------------
# InMemoryDriveClient — list_files
# ---------------------------------------------------------------------------


def test_list_files_returns_correct_count(populated_client: InMemoryDriveClient) -> None:
    files = populated_client.list_files(_FOLDER_A)
    assert len(files) == 2


def test_list_files_empty_folder(populated_client: InMemoryDriveClient) -> None:
    files = populated_client.list_files("nonexistent-folder")
    assert files == []


def test_list_files_metadata(populated_client: InMemoryDriveClient) -> None:
    files = populated_client.list_files(_FOLDER_A)
    names = {f.name for f in files}
    mimes = {f.mime_type for f in files}
    assert names == {"product.jpg", "logo.png"}
    assert "image/jpeg" in mimes
    assert "image/png" in mimes


def test_list_files_returns_drive_file_instances(
    populated_client: InMemoryDriveClient,
) -> None:
    files = populated_client.list_files(_FOLDER_A)
    for f in files:
        assert isinstance(f, DriveFile)
        assert f.file_id.startswith("mem-")


def test_list_files_size_matches_content(populated_client: InMemoryDriveClient) -> None:
    files = populated_client.list_files(_FOLDER_A)
    jpg = next(f for f in files if f.name == "product.jpg")
    # content is b"\xff\xd8\xff" + b"\x00" * 100 = 103 bytes
    assert jpg.size == 103


# ---------------------------------------------------------------------------
# InMemoryDriveClient — download_file
# ---------------------------------------------------------------------------


def test_download_file_round_trips_bytes(populated_client: InMemoryDriveClient) -> None:
    files = populated_client.list_files(_FOLDER_A)
    jpg = next(f for f in files if f.name == "product.jpg")
    data = populated_client.download_file(jpg.file_id)
    assert data[:3] == b"\xff\xd8\xff"
    assert len(data) == 103


def test_download_file_not_found_raises(populated_client: InMemoryDriveClient) -> None:
    with pytest.raises(DriveNotFoundError):
        populated_client.download_file("does-not-exist")


# ---------------------------------------------------------------------------
# InMemoryDriveClient — upload_file
# ---------------------------------------------------------------------------


def test_upload_file_returns_file_id(populated_client: InMemoryDriveClient) -> None:
    fid = populated_client.upload_file(
        folder_id=_FOLDER_A,
        name="new_video.mp4",
        mime_type="video/mp4",
        content=b"\x00" * 256,
    )
    assert isinstance(fid, str)
    assert fid.startswith("mem-")


def test_upload_file_appears_in_list(populated_client: InMemoryDriveClient) -> None:
    fid = populated_client.upload_file(
        folder_id=_FOLDER_A,
        name="new_video.mp4",
        mime_type="video/mp4",
        content=b"\x00" * 256,
    )
    files = populated_client.list_files(_FOLDER_A)
    ids = [f.file_id for f in files]
    assert fid in ids


def test_upload_file_downloadable(populated_client: InMemoryDriveClient) -> None:
    content = b"hello world"
    fid = populated_client.upload_file(
        folder_id=_FOLDER_B,
        name="readme.txt",
        mime_type="text/plain",
        content=content,
    )
    assert populated_client.download_file(fid) == content


def test_upload_to_new_folder(populated_client: InMemoryDriveClient) -> None:
    fid = populated_client.upload_file(
        folder_id="brand-new-folder",
        name="clip.mp4",
        mime_type="video/mp4",
        content=b"video",
    )
    files = populated_client.list_files("brand-new-folder")
    assert len(files) == 1
    assert files[0].file_id == fid


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_in_memory_client_satisfies_protocol() -> None:
    client = InMemoryDriveClient()
    assert isinstance(client, DriveClientProtocol)


# ---------------------------------------------------------------------------
# make_drive_client — fail-fast behaviour
# ---------------------------------------------------------------------------

_EXPECTED_MSG_FRAGMENT = (
    "Drive credentials not found at"
)


def test_make_drive_client_none_raises_auth_error() -> None:
    with pytest.raises(DriveAuthError) as exc_info:
        make_drive_client(None)
    msg = str(exc_info.value)
    assert "Drive credentials not found at" in msg
    assert "Drive integration is required" in msg
    assert "§16 Day 6" in msg
    assert "GOOGLE_DRIVE_CREDENTIALS_PATH" in msg


def test_make_drive_client_nonexistent_path_raises_auth_error(tmp_path: Path) -> None:
    creds = tmp_path / "nonexistent" / "service-account.json"
    with pytest.raises(DriveAuthError) as exc_info:
        make_drive_client(creds)
    msg = str(exc_info.value)
    assert "Drive credentials not found at" in msg
    assert "Drive integration is required" in msg
    assert "§16 Day 6" in msg
    assert "GOOGLE_DRIVE_CREDENTIALS_PATH" in msg


def test_make_drive_client_error_message_contains_path(tmp_path: Path) -> None:
    creds = tmp_path / "my-creds.json"
    # file does NOT exist
    with pytest.raises(DriveAuthError) as exc_info:
        make_drive_client(creds)
    assert str(creds) in str(exc_info.value)


def test_make_drive_client_error_for_none_contains_none_repr() -> None:
    with pytest.raises(DriveAuthError) as exc_info:
        make_drive_client(None)
    assert "None" in str(exc_info.value)
