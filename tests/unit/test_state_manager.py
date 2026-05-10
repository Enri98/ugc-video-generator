"""Unit tests for ugc_pipeline.state_manager (SPEC.md §7).

All tests use pytest's `tmp_path` fixture to avoid touching the real `state/` directory.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
import uuid

import pytest

from ugc_pipeline.models import ProductBrief, RunState, VideoSpec, VideoState
from ugc_pipeline.state_manager import (
    STALE_LOCK_SECONDS,
    acquire_lock,
    load_product_brief,
    load_run_state,
    load_video_spec,
    load_video_state,
    release_lock,
    save_product_brief,
    save_run_state,
    save_video_spec,
    save_video_state,
    sweep_locks,
    write_state_atomic,
)


# ---------------------------------------------------------------------------
# write_state_atomic
# ---------------------------------------------------------------------------


def test_write_state_atomic_creates_file(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "test.json"
    write_state_atomic(target, {"key": "value", "number": 42})
    assert target.is_file()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["key"] == "value"
    assert data["number"] == 42


def test_write_state_atomic_overwrites_existing(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "test.json"
    write_state_atomic(target, {"v": 1})
    write_state_atomic(target, {"v": 2})
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["v"] == 2


def test_write_state_atomic_no_temp_file_left_on_success(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "test.json"
    write_state_atomic(target, {"x": 1})
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], "No .tmp files should remain after a successful write"


def test_write_state_atomic_creates_parent_dirs(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "a" / "b" / "c.json"
    write_state_atomic(target, {"ok": True})
    assert target.is_file()


def test_write_state_atomic_datetime_serialised_as_string(tmp_path: pathlib.Path) -> None:
    """Datetime objects (which appear in model_dump output) are stringified via default=str."""
    from datetime import datetime, timezone

    target = tmp_path / "dt.json"
    write_state_atomic(target, {"ts": datetime.now(timezone.utc)})
    raw = target.read_text(encoding="utf-8")
    # The value must be a JSON string, not an object
    data = json.loads(raw)
    assert isinstance(data["ts"], str)


# ---------------------------------------------------------------------------
# VideoState round-trip
# ---------------------------------------------------------------------------


def _make_video_state(**kw: object) -> VideoState:
    defaults: dict = {
        "video_id": str(uuid.uuid4()),
        "product_id": "abc123def456",
        "spec_index": 0,
    }
    defaults.update(kw)
    return VideoState(**defaults)  # type: ignore[arg-type]


def test_video_state_round_trip(tmp_path: pathlib.Path) -> None:
    state = _make_video_state()
    save_video_state(state, root=tmp_path)
    loaded = load_video_state(state.video_id, root=tmp_path)
    assert loaded.video_id == state.video_id
    assert loaded.product_id == state.product_id
    assert loaded.status == "pending"


def test_video_state_round_trip_preserves_completed_steps(tmp_path: pathlib.Path) -> None:
    state = _make_video_state()
    state.completed_steps = ["first_frame_composite", "veo_generation"]
    save_video_state(state, root=tmp_path)
    loaded = load_video_state(state.video_id, root=tmp_path)
    assert loaded.completed_steps == ["first_frame_composite", "veo_generation"]


# ---------------------------------------------------------------------------
# VideoSpec round-trip
# ---------------------------------------------------------------------------


def _make_spec(**kw: object) -> VideoSpec:
    defaults: dict = {
        "video_id": str(uuid.uuid4()),
        "product_id": "abc123def456",
        "spec_index": 1,
        "tone": "energetic lifestyle",
        "narrative_arc": "From ordinary to extraordinary in 20 seconds.",
        "talent_id": "talent_01",
        "clip_count": 3,
        "scene_descriptions": ["A.", "B.", "C."],
        "script_blocks": ["", "Due.", ""],
    }
    defaults.update(kw)
    return VideoSpec(**defaults)  # type: ignore[arg-type]


def test_video_spec_round_trip(tmp_path: pathlib.Path) -> None:
    spec = _make_spec()
    save_video_spec(spec, root=tmp_path)
    loaded = load_video_spec(spec.video_id, root=tmp_path)
    assert loaded.video_id == spec.video_id
    assert loaded.tone == "energetic lifestyle"
    assert loaded.clip_count == 3


# ---------------------------------------------------------------------------
# ProductBrief round-trip
# ---------------------------------------------------------------------------


def _make_brief(**kw: object) -> ProductBrief:
    defaults: dict = {
        "product_id": "a3f9c12e7b04",
        "image_path": "artifacts/a3f9c12e7b04/product.jpg",
        "shape": "cylindrical mug",
        "dominant_colours": ["#F5F0E8"],
        "packaging_style": "kraft paper box",
        "inferred_category": "kitchenware",
        "lifestyle_contexts": ["morning routine"],
    }
    defaults.update(kw)
    return ProductBrief(**defaults)  # type: ignore[arg-type]


def test_product_brief_round_trip(tmp_path: pathlib.Path) -> None:
    brief = _make_brief()
    save_product_brief(brief, root=tmp_path)
    loaded = load_product_brief(brief.product_id, root=tmp_path)
    assert loaded.product_id == brief.product_id
    assert loaded.shape == "cylindrical mug"


# ---------------------------------------------------------------------------
# RunState round-trip
# ---------------------------------------------------------------------------


def test_run_state_round_trip(tmp_path: pathlib.Path) -> None:
    rs = RunState(run_id=str(uuid.uuid4()))
    rs.cumulative_cost_usd = 12.34
    save_run_state(rs, root=tmp_path)
    loaded = load_run_state(rs.run_id, root=tmp_path)
    assert loaded.run_id == rs.run_id
    assert abs(loaded.cumulative_cost_usd - 12.34) < 1e-9


# ---------------------------------------------------------------------------
# Lock acquire / release
# ---------------------------------------------------------------------------


def test_lock_acquire_creates_file(tmp_path: pathlib.Path) -> None:
    lock_path = acquire_lock("video-001", root=tmp_path)
    assert lock_path.is_file()
    # File should contain the current PID
    pid_text = lock_path.read_text(encoding="utf-8")
    assert pid_text == str(os.getpid())


def test_lock_release_removes_file(tmp_path: pathlib.Path) -> None:
    lock_path = acquire_lock("video-002", root=tmp_path)
    release_lock(lock_path)
    assert not lock_path.exists()


def test_lock_double_acquire_raises(tmp_path: pathlib.Path) -> None:
    acquire_lock("video-003", root=tmp_path)
    with pytest.raises(FileExistsError):
        acquire_lock("video-003", root=tmp_path)


def test_lock_release_idempotent(tmp_path: pathlib.Path) -> None:
    lock_path = acquire_lock("video-004", root=tmp_path)
    release_lock(lock_path)
    # Second release should not raise
    release_lock(lock_path)


# ---------------------------------------------------------------------------
# Stale lock detection
# ---------------------------------------------------------------------------


def test_stale_lock_is_removed_on_acquire(tmp_path: pathlib.Path) -> None:
    """A lock file with mtime older than STALE_LOCK_SECONDS is treated as abandoned."""
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    stale_path = lock_dir / "video-stale.lock"
    stale_path.write_text("99999", encoding="utf-8")

    # Back-date the mtime by more than the stale threshold
    old_time = time.time() - (STALE_LOCK_SECONDS + 10)
    os.utime(stale_path, (old_time, old_time))

    # acquire_lock should detect the stale file, remove it, and succeed
    lock_path = acquire_lock("video-stale", root=tmp_path)
    assert lock_path.is_file()
    release_lock(lock_path)


# ---------------------------------------------------------------------------
# sweep_locks
# ---------------------------------------------------------------------------


def test_sweep_locks_removes_all_locks(tmp_path: pathlib.Path) -> None:
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    (lock_dir / "a.lock").write_text("1")
    (lock_dir / "b.lock").write_text("2")
    (lock_dir / "keep.json").write_text("{}")  # should not be removed

    removed = sweep_locks(root=tmp_path)
    assert removed == 2
    assert not (lock_dir / "a.lock").exists()
    assert not (lock_dir / "b.lock").exists()
    assert (lock_dir / "keep.json").exists()


def test_sweep_locks_returns_zero_when_no_locks(tmp_path: pathlib.Path) -> None:
    removed = sweep_locks(root=tmp_path)
    assert removed == 0
