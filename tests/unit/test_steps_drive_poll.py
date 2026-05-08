"""Unit tests for src/ugc_pipeline/steps/drive_poll.py.

Covers:
- First poll: 2 images + 1 PDF → returns 2 PolledImages; PDF is skipped.
- Deduplication: pre-creating state/products/{id}.json causes that image to
  be skipped on second poll.
- Non-image MIME types trigger a WARNING log entry.
- compute_product_id is deterministic (SHA-256 prefix).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ugc_pipeline.steps.drive_poll import (
    PolledImage,
    compute_product_id,
    poll_drive_input,
)
from ugc_pipeline.utils.drive import InMemoryDriveClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FOLDER_INPUT = "drive-input-folder"

_JPG_BYTES = b"\xff\xd8\xff" + b"\xaa" * 120  # fake JPEG
_PNG_BYTES = b"\x89PNG\r\n" + b"\xbb" * 80     # fake PNG
_PDF_BYTES = b"%PDF-1.4" + b"\x00" * 50        # fake PDF

_JPG_PID = hashlib.sha256(_JPG_BYTES).hexdigest()[:12]
_PNG_PID = hashlib.sha256(_PNG_BYTES).hexdigest()[:12]


@pytest.fixture()
def drive_client() -> InMemoryDriveClient:
    """InMemoryDriveClient with 1 JPEG, 1 PNG, 1 PDF in the input folder."""
    return InMemoryDriveClient(
        initial={
            _FOLDER_INPUT: [
                ("product.jpg", "image/jpeg", _JPG_BYTES),
                ("banner.png", "image/png", _PNG_BYTES),
                ("datasheet.pdf", "application/pdf", _PDF_BYTES),
            ]
        }
    )


# ---------------------------------------------------------------------------
# compute_product_id
# ---------------------------------------------------------------------------


def test_compute_product_id_is_deterministic() -> None:
    data = b"some image bytes"
    assert compute_product_id(data) == compute_product_id(data)


def test_compute_product_id_length() -> None:
    assert len(compute_product_id(b"x" * 1000)) == 12


def test_compute_product_id_is_sha256_prefix() -> None:
    data = b"fixture"
    expected = hashlib.sha256(data).hexdigest()[:12]
    assert compute_product_id(data) == expected


def test_compute_product_id_differs_for_different_data() -> None:
    assert compute_product_id(b"a") != compute_product_id(b"b")


# ---------------------------------------------------------------------------
# poll_drive_input — first run
# ---------------------------------------------------------------------------


def test_first_poll_returns_two_images(
    drive_client: InMemoryDriveClient,
    tmp_path: Path,
) -> None:
    results = poll_drive_input(drive_client, _FOLDER_INPUT, state_root=tmp_path)
    assert len(results) == 2


def test_first_poll_all_are_polled_image(
    drive_client: InMemoryDriveClient,
    tmp_path: Path,
) -> None:
    results = poll_drive_input(drive_client, _FOLDER_INPUT, state_root=tmp_path)
    for r in results:
        assert isinstance(r, PolledImage)


def test_first_poll_product_ids(
    drive_client: InMemoryDriveClient,
    tmp_path: Path,
) -> None:
    results = poll_drive_input(drive_client, _FOLDER_INPUT, state_root=tmp_path)
    pids = {r.product_id for r in results}
    assert _JPG_PID in pids
    assert _PNG_PID in pids


def test_first_poll_image_bytes_correct(
    drive_client: InMemoryDriveClient,
    tmp_path: Path,
) -> None:
    results = poll_drive_input(drive_client, _FOLDER_INPUT, state_root=tmp_path)
    by_pid = {r.product_id: r for r in results}
    assert by_pid[_JPG_PID].image_bytes == _JPG_BYTES
    assert by_pid[_PNG_PID].image_bytes == _PNG_BYTES


def test_first_poll_pdf_skipped(
    drive_client: InMemoryDriveClient,
    tmp_path: Path,
) -> None:
    """PDF should not appear in results."""
    results = poll_drive_input(drive_client, _FOLDER_INPUT, state_root=tmp_path)
    filenames = {r.filename for r in results}
    assert "datasheet.pdf" not in filenames


# ---------------------------------------------------------------------------
# poll_drive_input — deduplication
# ---------------------------------------------------------------------------


def test_second_poll_skips_already_processed(
    drive_client: InMemoryDriveClient,
    tmp_path: Path,
) -> None:
    """Pre-seeding the products dir causes one image to be skipped."""
    products_dir = tmp_path / "products"
    products_dir.mkdir(parents=True, exist_ok=True)
    # Mark the JPEG as already processed
    (products_dir / f"{_JPG_PID}.json").write_text(
        json.dumps({"product_id": _JPG_PID}), encoding="utf-8"
    )

    results = poll_drive_input(drive_client, _FOLDER_INPUT, state_root=tmp_path)
    assert len(results) == 1
    assert results[0].product_id == _PNG_PID


def test_all_processed_returns_empty(
    drive_client: InMemoryDriveClient,
    tmp_path: Path,
) -> None:
    products_dir = tmp_path / "products"
    products_dir.mkdir(parents=True, exist_ok=True)
    for pid in (_JPG_PID, _PNG_PID):
        (products_dir / f"{pid}.json").write_text("{}", encoding="utf-8")

    results = poll_drive_input(drive_client, _FOLDER_INPUT, state_root=tmp_path)
    assert results == []


# ---------------------------------------------------------------------------
# poll_drive_input — WARNING for non-images
# ---------------------------------------------------------------------------


def test_non_image_triggers_warning(
    drive_client: InMemoryDriveClient,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """The PDF file must produce a drive_non_image_skipped log at WARNING level.

    structlog writes to stdout in test mode, so we capture stdout.
    """
    poll_drive_input(drive_client, _FOLDER_INPUT, state_root=tmp_path)
    captured = capsys.readouterr()
    # structlog emits the event name in the rendered output
    assert "drive_non_image_skipped" in captured.out


# ---------------------------------------------------------------------------
# poll_drive_input — empty folder
# ---------------------------------------------------------------------------


def test_empty_folder_returns_empty(tmp_path: Path) -> None:
    client = InMemoryDriveClient()
    results = poll_drive_input(client, "empty-folder", state_root=tmp_path)
    assert results == []


# ---------------------------------------------------------------------------
# poll_drive_input — custom mime filter
# ---------------------------------------------------------------------------


def test_custom_accepted_mimes(tmp_path: Path) -> None:
    """Only JPEG accepted; PNG is excluded."""
    client = InMemoryDriveClient(
        initial={
            "f": [
                ("a.jpg", "image/jpeg", _JPG_BYTES),
                ("b.png", "image/png", _PNG_BYTES),
            ]
        }
    )
    results = poll_drive_input(
        client,
        "f",
        state_root=tmp_path,
        accepted_mime_types=("image/jpeg",),
    )
    assert len(results) == 1
    assert results[0].product_id == _JPG_PID
