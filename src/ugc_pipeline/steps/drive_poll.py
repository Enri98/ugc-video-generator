"""Drive polling step for the UGC pipeline.

Implements SPEC.md §5 Step 1 and §10 Polling Algorithm.

Workflow:
1. List all files in the configured Drive input folder.
2. Skip non-image MIME types with a WARNING log.
3. Download each image and compute a deterministic product_id (SHA-256 prefix).
4. Skip images whose product_id already exists in state/products/.
5. Return a list of PolledImage objects for new images.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import structlog

from ugc_pipeline.utils.drive import DriveClientProtocol

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_product_id(image_bytes: bytes) -> str:
    """Return the 12-hex-char SHA-256 prefix of *image_bytes*.

    This deterministic identifier deduplicates product images across runs
    regardless of filename (SPEC.md §3 Stage 1).
    """
    return hashlib.sha256(image_bytes).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolledImage:
    """A new product image retrieved from Drive and ready for processing."""

    product_id: str
    image_bytes: bytes
    filename: str
    drive_file_id: str


# ---------------------------------------------------------------------------
# Main step function
# ---------------------------------------------------------------------------


def poll_drive_input(
    client: DriveClientProtocol,
    folder_id: str,
    *,
    state_root: Path,
    accepted_mime_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
    ),
) -> list[PolledImage]:
    """List the Drive input folder and return new, unprocessed product images.

    Parameters
    ----------
    client:
        Any object satisfying ``DriveClientProtocol`` (real or mock).
    folder_id:
        Drive folder ID to list.
    state_root:
        Root of the local state directory. Deduplication checks
        ``state_root/products/{product_id}.json``.
    accepted_mime_types:
        MIME types to accept. Files with other types are skipped with a
        WARNING log event (``drive_non_image_skipped``).

    Returns
    -------
    list[PolledImage]
        Images that are new (not yet in state/products/) and ready for
        the product_analyst step.
    """
    log.info("drive_poll_started", folder_id=folder_id)

    files = client.list_files(folder_id)
    log.debug("drive_poll_listed", folder_id=folder_id, file_count=len(files))

    products_dir = state_root / "products"
    results: list[PolledImage] = []

    for drive_file in files:
        # ------------------------------------------------------------------
        # Filter by MIME type
        # ------------------------------------------------------------------
        if drive_file.mime_type not in accepted_mime_types:
            log.warning(
                "drive_non_image_skipped",
                file_id=drive_file.file_id,
                name=drive_file.name,
                mime_type=drive_file.mime_type,
            )
            continue

        # ------------------------------------------------------------------
        # Download and deduplicate
        # ------------------------------------------------------------------
        image_bytes = client.download_file(drive_file.file_id)
        product_id = compute_product_id(image_bytes)

        state_file = products_dir / f"{product_id}.json"
        if state_file.exists():
            log.debug(
                "dedupe_skipped",
                product_id=product_id,
                drive_file_id=drive_file.file_id,
                name=drive_file.name,
            )
            continue

        # ------------------------------------------------------------------
        # New image — enqueue
        # ------------------------------------------------------------------
        results.append(
            PolledImage(
                product_id=product_id,
                image_bytes=image_bytes,
                filename=drive_file.name,
                drive_file_id=drive_file.file_id,
            )
        )
        log.info(
            "drive_poll_new_image",
            product_id=product_id,
            name=drive_file.name,
            drive_file_id=drive_file.file_id,
        )

    log.info(
        "drive_poll_completed",
        folder_id=folder_id,
        new_images=len(results),
        total_files=len(files),
    )
    return results
