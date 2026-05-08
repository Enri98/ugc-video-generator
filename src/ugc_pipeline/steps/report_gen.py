"""Report generation step for the UGC pipeline.

Implements SPEC.md §5 Step 10 and §4 Report contract.

Renders a human-readable TXT cost-and-metadata report for a completed (or
failed) video. Pure in-process Python; no external calls.

Idempotent: if the ``report`` artifact already exists with a non-zero size,
the step is skipped.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog

from ugc_pipeline.models import Report, RunState, VideoSpec, VideoState

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------

_DRY_RUN_HEADER = (
    "[DRY RUN]\n"
    "NOTE: Veo and Nano Banana costs below are estimates only.\n"
    "No paid media generation was performed in this run.\n"
    "========================================\n"
)


def render_report_text(
    report: Report,
    video_state: VideoState,
    video_spec: VideoSpec,
    *,
    dry_run: bool = False,
) -> str:
    """Render *report* as a plain-text string.

    Parameters
    ----------
    report:
        Fully-populated Report model (see SPEC.md §4).
    video_state:
        Mutable state for the video (provides status and artifact paths).
    video_spec:
        VideoSpec for the video (provides scene and script data).
    dry_run:
        When ``True``, prepends a ``[DRY RUN]`` header and a note that
        Veo/Nano Banana costs are estimates.

    Returns
    -------
    str
        The complete TXT report as a UTF-8 string.
    """
    lines: list[str] = []

    # ------------------------------------------------------------------
    # Dry-run header
    # ------------------------------------------------------------------
    if dry_run:
        lines.append(_DRY_RUN_HEADER)

    # ------------------------------------------------------------------
    # Header section
    # ------------------------------------------------------------------
    lines.append("UGC PIPELINE REPORT")
    lines.append("===================")
    lines.append(f"Run ID:     {report.run_id}")
    lines.append(f"Product ID: {report.product_id}")
    lines.append(f"Video ID:   {report.video_id}")
    lines.append(f"Spec Index: {report.spec_index}")
    lines.append(f"Talent:     {report.talent_id}")
    lines.append(f"Status:     {report.status}")
    lines.append(f"Generated:  {report.generated_at.isoformat()}")

    # ------------------------------------------------------------------
    # Scenes section
    # ------------------------------------------------------------------
    lines.append("")
    lines.append("SCENES")
    lines.append("------")
    scene_descs = report.scene_descriptions
    script_blks = report.script_blocks
    for i, (en, it) in enumerate(zip(scene_descs, script_blks), start=1):
        lines.append(f"[{i}] EN: {en}")
        lines.append(f"    IT: {it}")

    # ------------------------------------------------------------------
    # Token counts section
    # ------------------------------------------------------------------
    lines.append("")
    lines.append("TOKEN COUNTS")
    lines.append("------------")
    if report.token_counts:
        for step_name, count in report.token_counts.items():
            lines.append(f"{step_name}: {count}")
    else:
        lines.append("(not tracked in this run)")

    # ------------------------------------------------------------------
    # Costs section
    # ------------------------------------------------------------------
    costs = report.costs_usd
    lines.append("")
    lines.append("COSTS (USD)")
    lines.append("-----------")
    lines.append(f"product_analyst:    ${costs.product_analyst_usd:.4f}")
    lines.append(f"creative_director:  ${costs.creative_director_usd:.4f}")
    lines.append(f"first_frame:        ${costs.first_frame_usd:.4f}")
    lines.append(f"veo:                ${costs.veo_usd:.4f}")
    lines.append(f"safety_retry:       ${costs.safety_retry_usd:.4f}")
    lines.append(f"caption:            ${costs.caption_usd:.4f}")
    lines.append("----------------------------")
    lines.append(f"TOTAL:              ${costs.total:.4f}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Async step entry point
# ---------------------------------------------------------------------------


async def run_report_gen(
    video_state: VideoState,
    video_spec: VideoSpec,
    run_state: RunState,
    *,
    artifacts_root: Path,
    dry_run: bool = False,
) -> Path:
    """Render the TXT report and write it to disk.

    Parameters
    ----------
    video_state:
        Mutable VideoState; ``artifacts["report"]`` is set on success.
    video_spec:
        VideoSpec for the current video.
    run_state:
        RunState providing the ``run_id``.
    artifacts_root:
        Root directory under which per-video artifact subdirectories live.
    dry_run:
        When ``True``, the report includes a ``[DRY RUN]`` header.

    Returns
    -------
    Path
        Absolute path to the written TXT report.
    """
    t_start = time.monotonic()

    # ------------------------------------------------------------------
    # Idempotency check
    # ------------------------------------------------------------------
    existing_str = video_state.artifacts.get("report")
    if existing_str is not None:
        existing = Path(existing_str)
        if existing.exists() and existing.stat().st_size > 0:
            log.info(
                "step_skipped_idempotent",
                step="report_gen",
                video_id=video_state.video_id,
            )
            return existing

    log.info(
        "step_started",
        step="report_gen",
        video_id=video_state.video_id,
    )

    # ------------------------------------------------------------------
    # Build Report model
    # ------------------------------------------------------------------
    report = Report(
        run_id=run_state.run_id,
        product_id=video_state.product_id,
        video_id=video_state.video_id,
        talent_id=video_spec.talent_id,
        spec_index=video_spec.spec_index,
        scene_descriptions=video_spec.scene_descriptions,
        script_blocks=video_spec.script_blocks,
        token_counts={},  # v1: not tracked; populated by orchestrator if available
        costs_usd=video_state.costs_usd,
        status=video_state.status,
        generated_at=datetime.now(timezone.utc),
    )

    # ------------------------------------------------------------------
    # Render and write
    # ------------------------------------------------------------------
    text = render_report_text(report, video_state, video_spec, dry_run=dry_run)

    video_dir = artifacts_root / video_state.video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    out_path = video_dir / "report.txt"
    out_path.write_text(text, encoding="utf-8")

    video_state.artifacts["report"] = str(out_path)

    duration_ms = int((time.monotonic() - t_start) * 1000)
    log.info(
        "step_completed",
        step="report_gen",
        video_id=video_state.video_id,
        artifact_path=str(out_path),
        duration_ms=duration_ms,
    )

    return out_path
