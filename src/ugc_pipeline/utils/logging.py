"""structlog configuration for the UGC pipeline.

See SPEC.md §17 (Observability & Logging).

Design note (SPEC.md §17): JSON output goes to logs/pipeline.log via
RotatingFileHandler; human-readable output goes to stdout via StreamHandler.
Both handlers use the same structlog chain (JSONRenderer globally), which
means stdout also emits JSON. This is the simplest consistent approach — a
branching processor chain adds complexity without meaningful benefit for
this single-operator tool. The distinction in §17 is preserved by comment.

Idempotency: a module-level ``_configured`` flag prevents double-setup when
``configure_logging`` is called twice in the same process (e.g. in test
suites that call it in a fixture and then again in the function under test).
"""

from __future__ import annotations

import logging
import pathlib
from logging.handlers import RotatingFileHandler
from typing import Any

import structlog

# Module-level flag — configure_logging is a no-op on the second call.
_configured: bool = False

# Module-level convenience alias so callers can do:
#   from ugc_pipeline.utils.logging import get_logger
#   log = get_logger(__name__)
get_logger = structlog.get_logger


def configure_logging(
    log_file: str = "logs/pipeline.log",
    level: str = "INFO",
    max_bytes: int = 10_485_760,  # 10 MB
    backup_count: int = 5,
) -> None:
    """Configure structlog + stdlib logging for the pipeline.

    Per SPEC.md §17:
    - JSON lines written to *log_file* via RotatingFileHandler.
    - Human-readable (JSON) written to stdout via StreamHandler.
    - Log rotation: *max_bytes* per file, *backup_count* rotated copies.

    This function is idempotent: subsequent calls in the same process are
    no-ops (guarded by the module-level ``_configured`` flag).
    """
    global _configured
    if _configured:
        return

    # Ensure the log directory exists.
    log_path = pathlib.Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Rotating file handler — JSON log lines.
    rotating_handler = RotatingFileHandler(
        str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    # structlog pre-formats the message; stdlib formatter just passes it through.
    rotating_handler.setFormatter(logging.Formatter("%(message)s"))

    # Stream handler — also JSON (stdout) per the simplest consistent approach.
    # Explicitly pass sys.stdout so log lines appear on stdout (not stderr),
    # which is where pytest capsys captures output and where §17 expects it.
    import sys
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    # Configure the root logger level without using force=True (which would
    # remove pytest's capfd handler in test environments).
    root = logging.getLogger()
    root.setLevel(level)
    # Add our handlers only if they have not already been added.
    handler_types = {type(h) for h in root.handlers}
    if RotatingFileHandler not in handler_types:
        root.addHandler(rotating_handler)
    # Always add our stream handler so log lines reach stdout.
    # We do not clear existing handlers to preserve test capture infrastructure.
    existing_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        for h in root.handlers
    )
    if not existing_stream:
        root.addHandler(stream_handler)

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),  # JSON to file and stdout
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configured = True


def log_step_event(
    log: Any,
    event: str,
    state: Any,  # VideoState — typed as Any to avoid circular imports
    step_name: str,
    **kwargs: Any,
) -> None:
    """Emit a structured log event aligned with SPEC.md §17 prompt_versions contract.

    Reads ``state.prompt_versions[step_name]`` and emits it under the key
    ``prompt_versions`` (plural — never singular, per SPEC.md §4 alignment note).

    Parameters
    ----------
    log:
        A structlog bound logger instance.
    event:
        Machine-readable event name (e.g. "step_completed").
    state:
        VideoState instance for the active video.
    step_name:
        Name of the pipeline step (e.g. "veo_generation").
    **kwargs:
        Additional key/value pairs merged into the log record.
    """
    pv = state.prompt_versions.get(step_name)
    log.info(
        event,
        step=step_name,
        video_id=state.video_id,
        product_id=state.product_id,
        # SPEC.md §4 alignment: always "prompt_versions" (plural)
        prompt_versions={step_name: pv.model_dump(mode="json") if pv else None},
        **kwargs,
    )
