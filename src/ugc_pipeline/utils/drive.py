"""Google Drive client abstractions for the UGC pipeline.

Implements SPEC.md §10 Drive Integration.

Three layers:
- ``DriveClientProtocol`` — structural typing contract (sync).
- ``InMemoryDriveClient`` — in-memory test double; never hits the network.
- ``GoogleDriveClient`` — real implementation via google-api-python-client
  with tenacity retry for transient HTTP errors.

Factory:
- ``make_drive_client`` — returns ``InMemoryDriveClient`` when credentials are
  absent (test only) or ``GoogleDriveClient`` when credentials exist. Raises
  ``DriveAuthError`` when the path is explicitly ``None`` or the file does not
  exist (fail-fast per §10).

Sync note: Drive SDK calls are synchronous. The orchestrator wraps them with
``asyncio.to_thread`` where needed (see ``drive_upload.py``).
"""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DriveError(Exception):
    """Base class for Drive-related errors."""


class DriveAuthError(DriveError):
    """Raised when Drive authentication fails or credentials are missing."""


class DriveNotFoundError(DriveError):
    """Raised when a requested file or folder is not found in Drive."""


class DriveTransientError(DriveError):
    """Raised for transient Drive API errors (network, 5xx); eligible for retry."""


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriveFile:
    """Metadata for a single file returned by the Drive API."""

    file_id: str
    name: str
    mime_type: str
    size: int


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DriveClientProtocol(Protocol):
    """Structural typing contract for Drive clients.

    All methods are synchronous. The async orchestrator wraps them with
    ``asyncio.to_thread``.
    """

    def list_files(self, folder_id: str) -> list[DriveFile]:
        """List all non-trashed files directly under *folder_id*."""
        ...

    def download_file(self, file_id: str) -> bytes:
        """Download and return the full byte content of *file_id*."""
        ...

    def upload_file(
        self,
        *,
        folder_id: str,
        name: str,
        mime_type: str,
        content: bytes,
    ) -> str:
        """Upload *content* as a new file and return the new ``file_id``."""
        ...


# ---------------------------------------------------------------------------
# In-memory test double
# ---------------------------------------------------------------------------


class InMemoryDriveClient:
    """In-memory implementation of DriveClientProtocol for unit tests.

    Parameters
    ----------
    initial:
        Optional seed data. Keys are folder IDs; values are lists of
        ``(name, mime_type, content)`` tuples. File IDs are auto-generated
        as ``"mem-<counter>"``.
    """

    _counter = itertools.count(1)

    def __init__(
        self,
        initial: dict[str, list[tuple[str, str, bytes]]] | None = None,
    ) -> None:
        # _files maps file_id -> {"meta": DriveFile, "content": bytes, "folder_id": str}
        self._files: dict[str, dict] = {}
        # _folders maps folder_id -> list[file_id]
        self._folders: dict[str, list[str]] = {}

        if initial:
            for folder_id, entries in initial.items():
                for name, mime_type, content in entries:
                    self._add_file(folder_id, name, mime_type, content)

    def _add_file(
        self, folder_id: str, name: str, mime_type: str, content: bytes
    ) -> str:
        file_id = f"mem-{next(self._counter)}"
        meta = DriveFile(
            file_id=file_id,
            name=name,
            mime_type=mime_type,
            size=len(content),
        )
        self._files[file_id] = {
            "meta": meta,
            "content": content,
            "folder_id": folder_id,
        }
        self._folders.setdefault(folder_id, []).append(file_id)
        return file_id

    # ------------------------------------------------------------------
    # DriveClientProtocol implementation
    # ------------------------------------------------------------------

    def list_files(self, folder_id: str) -> list[DriveFile]:
        """Return metadata for all files in *folder_id*."""
        fids = self._folders.get(folder_id, [])
        return [self._files[fid]["meta"] for fid in fids if fid in self._files]

    def download_file(self, file_id: str) -> bytes:
        """Return the byte content of *file_id*."""
        if file_id not in self._files:
            raise DriveNotFoundError(f"File not found: {file_id!r}")
        return self._files[file_id]["content"]

    def upload_file(
        self,
        *,
        folder_id: str,
        name: str,
        mime_type: str,
        content: bytes,
    ) -> str:
        """Add *content* to the in-memory store and return the new file_id."""
        return self._add_file(folder_id, name, mime_type, content)


# ---------------------------------------------------------------------------
# Real Google Drive client
# ---------------------------------------------------------------------------


class GoogleDriveClient:
    """Production Drive client using google-api-python-client.

    TODO Day 8: validate against a real Drive folder with a sample image
    and verify upload round-trip before enabling in production.

    Transient HTTP errors (5xx, connection errors) are retried up to 3 times
    using tenacity with exponential back-off (base 2 s).

    Parameters
    ----------
    credentials_path:
        Path to a service-account JSON key file.
    """

    def __init__(self, credentials_path: Path) -> None:
        from google.oauth2 import service_account  # type: ignore[import-untyped]
        from googleapiclient.discovery import build  # type: ignore[import-untyped]

        _SCOPES = ["https://www.googleapis.com/auth/drive"]
        creds = service_account.Credentials.from_service_account_file(
            str(credentials_path), scopes=_SCOPES
        )
        self._service = build("drive", "v3", credentials=creds, num_retries=0)
        log.info("drive_client_initialised", credentials_path=str(credentials_path))

    def _retry(self):  # type: ignore[no-untyped-def]
        """Return a tenacity Retry decorator for Drive calls."""
        from tenacity import (  # type: ignore[import-untyped]
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )
        from googleapiclient.errors import HttpError  # type: ignore[import-untyped]

        return retry(
            retry=retry_if_exception_type(HttpError),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )

    def list_files(self, folder_id: str) -> list[DriveFile]:
        """List all non-trashed files directly under *folder_id*."""
        try:
            retry_deco = self._retry()

            @retry_deco
            def _call() -> list[DriveFile]:
                results = (
                    self._service.files()
                    .list(
                        q=f"'{folder_id}' in parents and trashed = false",
                        fields="files(id, name, mimeType, size)",
                        pageSize=1000,
                    )
                    .execute()
                )
                files = []
                for f in results.get("files", []):
                    files.append(
                        DriveFile(
                            file_id=f["id"],
                            name=f["name"],
                            mime_type=f["mimeType"],
                            size=int(f.get("size", 0)),
                        )
                    )
                return files

            return _call()
        except Exception as exc:
            from googleapiclient.errors import HttpError  # type: ignore[import-untyped]

            if isinstance(exc, HttpError) and exc.status_code in (401, 403):
                raise DriveAuthError(str(exc)) from exc
            if isinstance(exc, HttpError) and exc.status_code == 404:
                raise DriveNotFoundError(folder_id) from exc
            raise DriveTransientError(str(exc)) from exc

    def download_file(self, file_id: str) -> bytes:
        """Download and return the byte content of *file_id*."""
        try:
            from googleapiclient.http import MediaIoBaseDownload  # type: ignore[import-untyped]
            import io

            retry_deco = self._retry()

            @retry_deco
            def _call() -> bytes:
                req = self._service.files().get_media(fileId=file_id)
                buf = io.BytesIO()
                downloader = MediaIoBaseDownload(buf, req)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                return buf.getvalue()

            return _call()
        except Exception as exc:
            from googleapiclient.errors import HttpError  # type: ignore[import-untyped]

            if isinstance(exc, HttpError) and exc.status_code == 404:
                raise DriveNotFoundError(file_id) from exc
            raise DriveTransientError(str(exc)) from exc

    def upload_file(
        self,
        *,
        folder_id: str,
        name: str,
        mime_type: str,
        content: bytes,
    ) -> str:
        """Upload *content* as a new file and return the Drive file ID."""
        try:
            from googleapiclient.http import MediaInMemoryUpload  # type: ignore[import-untyped]

            retry_deco = self._retry()

            @retry_deco
            def _call() -> str:
                media = MediaInMemoryUpload(content, mimetype=mime_type, resumable=False)
                metadata = {"name": name, "parents": [folder_id]}
                result = (
                    self._service.files()
                    .create(body=metadata, media_body=media, fields="id")
                    .execute()
                )
                return str(result["id"])

            return _call()
        except Exception as exc:
            from googleapiclient.errors import HttpError  # type: ignore[import-untyped]

            if isinstance(exc, HttpError) and exc.status_code in (401, 403):
                raise DriveAuthError(str(exc)) from exc
            raise DriveTransientError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_AUTH_ERROR_TEMPLATE = (
    "Drive credentials not found at '{path}'.\n"
    "Drive integration is required for input polling and output upload.\n"
    "See §16 Day 6 in SPEC.md for the service account bootstrap procedure.\n"
    "Set GOOGLE_DRIVE_CREDENTIALS_PATH in .env once credentials are in place."
)


def make_drive_client(credentials_path: Path | None) -> DriveClientProtocol:
    """Return a ``GoogleDriveClient`` for *credentials_path*.

    Raises ``DriveAuthError`` with the canonical SPEC.md §10 message if
    *credentials_path* is ``None`` or the file does not exist.

    Parameters
    ----------
    credentials_path:
        Absolute path to a service-account JSON key file, or ``None``.
    """
    display_path = str(credentials_path) if credentials_path is not None else "None"
    if credentials_path is None or not credentials_path.exists():
        raise DriveAuthError(_AUTH_ERROR_TEMPLATE.format(path=display_path))
    return GoogleDriveClient(credentials_path)
