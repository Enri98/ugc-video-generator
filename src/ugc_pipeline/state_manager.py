"""Atomic state persistence and file-lock helpers for the UGC pipeline.

All read/write operations target the `state/` directory tree described in SPEC.md §7.
Writes use a tempfile-then-replace pattern to guarantee atomicity on Windows (same
volume) and POSIX (rename syscall).

Lock staleness policy: if a `.lock` file exists and its mtime is older than
STALE_LOCK_SECONDS (600 s), the lock is treated as abandoned and removed. This is
simpler and more portable than process-table inspection.

Computed-field exclusion note: `CostBreakdown.total` is a `@computed_field`. Pydantic
includes it in `model_dump()` output but rejects it on `model_validate()` because
`extra="forbid"` is set. We therefore exclude all computed fields at serialization time
by calling `model.model_dump(mode="json", exclude={"costs_usd": {"total"}})` for models
that embed `CostBreakdown`, and by using a small helper that strips computed fields from
arbitrary nested dicts before writing.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import time
from typing import Any

from pydantic import BaseModel

from ugc_pipeline.models import (
    ProductBrief,
    RunState,
    VideoSpec,
    VideoState,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STALE_LOCK_SECONDS: int = 600  # locks older than this are treated as stale


# ---------------------------------------------------------------------------
# Computed-field exclusion helper
# ---------------------------------------------------------------------------


def _dump_model(model: BaseModel) -> dict[str, Any]:
    """Serialise *model* to a JSON-safe dict, stripping computed fields at all levels.

    Pydantic v2's `model_dump(mode="json")` includes `@computed_field` values.
    Loading such a dict back through `model_validate` on a model with
    `extra="forbid"` raises a `ValidationError`. This helper strips every key
    that corresponds to a computed field, recursing into nested dicts that
    represent nested Pydantic models.
    """

    def _strip(data: Any, cls: type) -> Any:  # type: ignore[return]
        if not isinstance(data, dict):
            return data
        computed = set(getattr(cls, "model_computed_fields", {}).keys())
        result: dict[str, Any] = {}
        for key, value in data.items():
            if key in computed:
                continue
            # Resolve the nested model class (if any) from field annotations
            nested_cls: type | None = None
            fields = getattr(cls, "model_fields", {})
            if key in fields:
                ann = fields[key].annotation
                # Unwrap Optional[X] (union with None)
                origin = getattr(ann, "__origin__", None)
                import types as _types
                if origin is _types.UnionType:
                    args = [a for a in ann.__args__ if a is not type(None)]
                    ann = args[0] if args else ann
                if isinstance(ann, type) and issubclass(ann, BaseModel):
                    nested_cls = ann
            if nested_cls is not None and isinstance(value, dict):
                result[key] = _strip(value, nested_cls)
            else:
                result[key] = value
        return result

    raw = model.model_dump(mode="json")
    return _strip(raw, type(model))


# ---------------------------------------------------------------------------
# Core atomic write  (SPEC.md §7 "Atomic Write Protocol")
# ---------------------------------------------------------------------------


def write_state_atomic(path: pathlib.Path, data: dict[str, Any]) -> None:
    """Write *data* as indented JSON to *path* atomically.

    The temp file is placed in the same directory as *path* to guarantee same-volume
    placement on Windows, making `os.replace` atomic (or best-effort on Windows
    same-volume when the target exists).
    """
    dir_ = path.parent
    dir_.mkdir(parents=True, exist_ok=True)
    # fd opened in same directory to guarantee same-volume placement on Windows
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_path, path)  # atomic on POSIX; best-effort on Windows same-volume
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# VideoState persistence
# ---------------------------------------------------------------------------


def _video_state_path(video_id: str, root: pathlib.Path) -> pathlib.Path:
    return root / "videos" / f"{video_id}.state.json"


def save_video_state(state: VideoState, root: pathlib.Path = pathlib.Path("state")) -> None:
    """Persist *state* to `state/videos/{video_id}.state.json`."""
    path = _video_state_path(state.video_id, root)
    write_state_atomic(path, _dump_model(state))


def load_video_state(video_id: str, root: pathlib.Path = pathlib.Path("state")) -> VideoState:
    """Load and validate VideoState from disk."""
    path = _video_state_path(video_id, root)
    data = json.loads(path.read_text(encoding="utf-8"))
    return VideoState.model_validate(data)


# ---------------------------------------------------------------------------
# VideoSpec persistence
# ---------------------------------------------------------------------------


def _video_spec_path(video_id: str, root: pathlib.Path) -> pathlib.Path:
    return root / "videos" / f"{video_id}.spec.json"


def save_video_spec(spec: VideoSpec, root: pathlib.Path = pathlib.Path("state")) -> None:
    """Persist *spec* to `state/videos/{video_id}.spec.json`."""
    path = _video_spec_path(spec.video_id, root)
    write_state_atomic(path, _dump_model(spec))


def load_video_spec(video_id: str, root: pathlib.Path = pathlib.Path("state")) -> VideoSpec:
    """Load and validate VideoSpec from disk."""
    path = _video_spec_path(video_id, root)
    data = json.loads(path.read_text(encoding="utf-8"))
    return VideoSpec.model_validate(data)


# ---------------------------------------------------------------------------
# ProductBrief persistence
# ---------------------------------------------------------------------------


def _product_brief_path(product_id: str, root: pathlib.Path) -> pathlib.Path:
    return root / "products" / f"{product_id}.json"


def save_product_brief(
    brief: ProductBrief, root: pathlib.Path = pathlib.Path("state")
) -> None:
    """Persist *brief* to `state/products/{product_id}.json`."""
    path = _product_brief_path(brief.product_id, root)
    write_state_atomic(path, _dump_model(brief))


def load_product_brief(
    product_id: str, root: pathlib.Path = pathlib.Path("state")
) -> ProductBrief:
    """Load and validate ProductBrief from disk."""
    path = _product_brief_path(product_id, root)
    data = json.loads(path.read_text(encoding="utf-8"))
    return ProductBrief.model_validate(data)


# ---------------------------------------------------------------------------
# RunState persistence
# ---------------------------------------------------------------------------


def _run_state_path(run_id: str, root: pathlib.Path) -> pathlib.Path:
    return root / "runs" / f"{run_id}.json"


def save_run_state(
    state: RunState, root: pathlib.Path = pathlib.Path("state")
) -> None:
    """Persist *state* to `state/runs/{run_id}.json`."""
    path = _run_state_path(state.run_id, root)
    write_state_atomic(path, _dump_model(state))


def load_run_state(
    run_id: str, root: pathlib.Path = pathlib.Path("state")
) -> RunState:
    """Load and validate RunState from disk."""
    path = _run_state_path(run_id, root)
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunState.model_validate(data)


# ---------------------------------------------------------------------------
# File-lock helpers  (SPEC.md §7 "File-Lock Convention")
# ---------------------------------------------------------------------------


def _lock_path(video_id: str, root: pathlib.Path) -> pathlib.Path:
    return root / "locks" / f"{video_id}.lock"


def acquire_lock(video_id: str, root: pathlib.Path = pathlib.Path("state")) -> pathlib.Path:
    """Acquire an exclusive file lock for *video_id*.

    Creates `state/locks/{video_id}.lock` containing the current PID. If a lock
    file already exists and its mtime is older than STALE_LOCK_SECONDS, it is
    treated as stale and removed before re-acquiring.

    Raises `FileExistsError` if the lock is held by a live process.
    """
    lock_dir = root / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = _lock_path(video_id, root)

    # Remove stale lock if older than threshold
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age > STALE_LOCK_SECONDS:
            # Lock file is older than STALE_LOCK_SECONDS — treat as abandoned
            try:
                path.unlink()
            except OSError:
                pass

    # Atomically create the lock file; raises FileExistsError if it exists
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise FileExistsError(
            f"Lock for video {video_id!r} is held by another process. "
            f"Lock path: {path}"
        ) from None

    try:
        os.write(fd, str(os.getpid()).encode())
    finally:
        os.close(fd)

    return path


def release_lock(lock_path: pathlib.Path) -> None:
    """Remove the lock file at *lock_path*, ignoring missing-file errors."""
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Lock sweep  (called at orchestrator startup)
# ---------------------------------------------------------------------------


def sweep_locks(root: pathlib.Path = pathlib.Path("state")) -> int:
    """Remove all `*.lock` files under `state/locks/`. Returns the count removed.

    Called once at orchestrator startup to clean up locks left by a prior crashed run.
    """
    lock_dir = root / "locks"
    if not lock_dir.is_dir():
        return 0
    removed = 0
    for lock_file in lock_dir.glob("*.lock"):
        try:
            lock_file.unlink()
            removed += 1
        except OSError:
            pass
    return removed
