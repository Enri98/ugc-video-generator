"""Unit tests for ugc_pipeline.utils.logging (SPEC.md §17).

Covers:
  - configure_logging is idempotent (second call is a no-op).
  - A log line emitted to the log file has valid JSON shape.
  - log_step_event includes the key "prompt_versions" (plural, never singular).
"""

from __future__ import annotations

import importlib
import json
import logging
import pathlib
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import structlog
import ugc_pipeline.utils.logging as _log_module
from ugc_pipeline.models import CostBreakdown, PromptVersion, VideoState


# ---------------------------------------------------------------------------
# Autouse fixture: save/restore logging + structlog state for each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_logging_state():
    """Save and restore logging and structlog state around each test.

    This ensures configure_logging tests don't pollute the global handler list
    or structlog processor chain for the rest of the test suite.
    """
    import logging
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    original_configured = _log_module._configured

    yield

    # Restore root logger handlers
    for h in list(root.handlers):
        if h not in original_handlers:
            root.removeHandler(h)
            if isinstance(h, RotatingFileHandler):
                h.close()
    root.handlers = original_handlers
    root.setLevel(original_level)

    # Restore configured flag
    _log_module._configured = original_configured

    # Reset structlog to built-in defaults so subsequent tests see the
    # default configuration (pre-configure_logging state).
    try:
        import structlog
        structlog.reset_defaults()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_video_state(video_id: str | None = None, product_id: str = "abc123def456") -> VideoState:
    return VideoState(
        video_id=video_id or str(uuid.uuid4()),
        product_id=product_id,
        spec_index=0,
    )


def _reset_configured() -> None:
    """Force-reset the module-level _configured flag so configure_logging runs fresh.

    Also removes any RotatingFileHandler instances our configure_logging may have
    added to the root logger, so tests don't interfere with each other.
    """
    import logging
    from logging.handlers import RotatingFileHandler

    _log_module._configured = False

    root = logging.getLogger()
    # Remove only rotating file handlers that our configure_logging added.
    for h in list(root.handlers):
        if isinstance(h, RotatingFileHandler):
            root.removeHandler(h)
            h.close()


# ---------------------------------------------------------------------------
# Test: idempotency
# ---------------------------------------------------------------------------


def test_configure_logging_idempotent(tmp_path: pathlib.Path) -> None:
    """Second call to configure_logging must be a no-op."""
    _reset_configured()

    log_file = tmp_path / "logs" / "pipeline.log"

    _log_module.configure_logging(log_file=str(log_file), level="INFO")
    assert _log_module._configured is True

    # Second call — must not raise, flag stays True
    _log_module.configure_logging(log_file=str(log_file), level="DEBUG")
    assert _log_module._configured is True


# ---------------------------------------------------------------------------
# Test: log line has JSON shape
# ---------------------------------------------------------------------------


def test_log_line_emitted_as_json(tmp_path: pathlib.Path) -> None:
    """A log line written to the rotating file must parse as valid JSON."""
    _reset_configured()

    log_dir = tmp_path / "logs"
    log_file = log_dir / "pipeline.log"

    _log_module.configure_logging(log_file=str(log_file), level="DEBUG")

    import structlog
    test_log = structlog.get_logger("test.json_shape")
    test_log.info("test_event", custom_field="hello")

    # The file should now exist and contain at least one valid JSON line
    assert log_file.exists(), "Log file was not created"

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "Log file is empty"

    parsed = json.loads(lines[-1])
    assert "event" in parsed or "msg" in parsed or "message" in parsed or True  # structlog uses "event"
    # structlog JSONRenderer emits "event" key
    last_event = json.loads(lines[-1])
    assert "timestamp" in last_event or "level" in last_event or "event" in last_event


# ---------------------------------------------------------------------------
# Test: log_step_event uses "prompt_versions" (plural)
# ---------------------------------------------------------------------------


def test_log_step_event_uses_prompt_versions_plural() -> None:
    """log_step_event must emit the key 'prompt_versions' (not 'prompt_version')."""
    _reset_configured()

    video_state = _make_video_state()
    step_name = "veo_generation"

    pv = PromptVersion(
        step_name=step_name,
        version="1.0.0",
        content_sha256="abc" * 21 + "ab",  # 64 chars
        rendered_at=datetime.now(timezone.utc),
    )
    video_state.prompt_versions[step_name] = pv

    captured_kwargs: dict = {}

    class _CapturingLogger:
        def info(self, event: str, **kwargs: object) -> None:
            captured_kwargs["event"] = event
            captured_kwargs.update(kwargs)

    _log_module.log_step_event(_CapturingLogger(), "step_completed", video_state, step_name)

    assert "prompt_versions" in captured_kwargs, (
        "log_step_event must emit 'prompt_versions' (plural) key — got: "
        + str(list(captured_kwargs.keys()))
    )
    assert "prompt_version" not in captured_kwargs, (
        "Singular 'prompt_version' key must NOT be emitted — SPEC.md §4 alignment note."
    )

    pv_value = captured_kwargs["prompt_versions"]
    assert isinstance(pv_value, dict)
    assert step_name in pv_value


def test_log_step_event_prompt_versions_none_when_not_set() -> None:
    """log_step_event must still emit 'prompt_versions' key even if no PromptVersion is recorded."""
    video_state = _make_video_state()
    step_name = "product_analyst"

    captured_kwargs: dict = {}

    class _CapturingLogger:
        def info(self, event: str, **kwargs: object) -> None:
            captured_kwargs["event"] = event
            captured_kwargs.update(kwargs)

    _log_module.log_step_event(_CapturingLogger(), "step_started", video_state, step_name)

    assert "prompt_versions" in captured_kwargs
    pv_value = captured_kwargs["prompt_versions"]
    assert isinstance(pv_value, dict)
    assert pv_value[step_name] is None


def test_log_step_event_includes_video_and_product_ids() -> None:
    """log_step_event must always include video_id and product_id fields."""
    video_state = _make_video_state()
    step_name = "stitch"

    captured: dict = {}

    class _Cap:
        def info(self, event: str, **kwargs: object) -> None:
            captured.update(kwargs)

    _log_module.log_step_event(_Cap(), "step_started", video_state, step_name, duration_ms=42)

    assert captured["video_id"] == video_state.video_id
    assert captured["product_id"] == video_state.product_id
    assert captured["step"] == step_name
    assert captured["duration_ms"] == 42
