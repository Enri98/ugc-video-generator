"""Pydantic v2 data models for the UGC video pipeline.

All models are defined per SPEC.md §4. Use `model_config = ConfigDict(extra="forbid")`
to catch unexpected fields early. UTC timestamps are generated via `datetime.now(timezone.utc)`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

VideoStatus = Literal[
    "pending",
    "in_progress",
    "completed",
    "completed_local",  # completed without Drive upload (credentials not configured)
    "failed_safety",
    "failed_budget",
    "failed_error",
    "failed_timeout",
]


# ---------------------------------------------------------------------------
# PromptVersion  (SPEC.md §4)
# ---------------------------------------------------------------------------


class PromptVersion(BaseModel):
    """Records which prompt template version was used for a given pipeline step."""

    model_config = ConfigDict(extra="forbid")

    step_name: str
    version: str
    content_sha256: str
    rendered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# CostBreakdown  (SPEC.md §4)
# ---------------------------------------------------------------------------


class CostBreakdown(BaseModel):
    """Running cost breakdown for a single video.

    All monetary values are in USD. `total` is a computed property that sums
    all per-step fields so it is always consistent.
    """

    model_config = ConfigDict(extra="forbid")

    product_analyst_usd: float = 0.0
    creative_director_usd: float = 0.0
    first_frame_usd: float = 0.0
    veo_usd: float = 0.0
    safety_retry_usd: float = 0.0
    caption_usd: float = 0.0  # faster-whisper is local/free; reserved for future use

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> float:
        """Sum of all per-step cost fields."""
        return (
            self.product_analyst_usd
            + self.creative_director_usd
            + self.first_frame_usd
            + self.veo_usd
            + self.safety_retry_usd
            + self.caption_usd
        )


# ---------------------------------------------------------------------------
# ProductBrief  (SPEC.md §4)
# ---------------------------------------------------------------------------


class ProductBrief(BaseModel):
    """Structured attributes extracted from a product image by the product analyst step.

    `product_id` is a 12-hex-char SHA-256 prefix of the raw image bytes.
    """

    model_config = ConfigDict(extra="forbid")

    product_id: str
    image_path: str
    shape: str
    dominant_colours: list[str]
    packaging_style: str
    inferred_category: str
    lifestyle_contexts: list[str]
    visual_notes: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# VideoSpec  (SPEC.md §4)
# ---------------------------------------------------------------------------


class VideoSpec(BaseModel):
    """Creative specification for one video, produced by the creative director step.

    `script_blocks` contain Italian voiceover text (one entry per clip).
    `scene_descriptions` are English cinematographic descriptions (one per clip).
    """

    model_config = ConfigDict(extra="forbid")

    video_id: str
    product_id: str
    spec_index: int
    tone: str
    narrative_arc: str
    talent_id: str
    clip_count: int
    scene_descriptions: list[str]
    script_blocks: list[str]
    visual_style_notes: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# VideoState  (SPEC.md §4)
# ---------------------------------------------------------------------------


class VideoState(BaseModel):
    """Mutable runtime state for a single video, persisted after every step.

    Updated at `state/videos/{video_id}.state.json`.
    """

    model_config = ConfigDict(extra="forbid")

    video_id: str
    product_id: str
    spec_index: int
    status: VideoStatus = "pending"
    current_step: str | None = None
    completed_steps: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    costs_usd: CostBreakdown = Field(default_factory=CostBreakdown)
    prompt_versions: dict[str, PromptVersion] = Field(default_factory=dict)
    last_error: str | None = None
    drive_video_file_id: str | None = None
    drive_report_file_id: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# RunState  (SPEC.md §4)
# ---------------------------------------------------------------------------


class RunState(BaseModel):
    """Global state for one pipeline run, persisted at `state/runs/{run_id}.json`."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ended_at: datetime | None = None
    cumulative_cost_usd: float = 0.0
    kill_switch_fired: bool = False
    products_seen: list[str] = Field(default_factory=list)
    videos_completed: int = 0
    videos_failed: int = 0


# ---------------------------------------------------------------------------
# Report  (SPEC.md §4)
# ---------------------------------------------------------------------------


class Report(BaseModel):
    """Plain-text report metadata for one completed (or failed) video."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    product_id: str
    video_id: str
    talent_id: str
    spec_index: int
    scene_descriptions: list[str]
    script_blocks: list[str]
    token_counts: dict[str, int]
    costs_usd: CostBreakdown
    status: str
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
