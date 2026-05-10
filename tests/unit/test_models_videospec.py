"""Unit tests for VideoSpec field additions and model_validator.

Tests cover:
- speaking_clip_index default value
- Happy-path validation
- All four validator constraints: scene/script length parity, index range, speech placement
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from ugc_pipeline.models import VideoSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(**overrides: object) -> VideoSpec:
    """Return a minimal valid VideoSpec for a ceramic coffee cup fixture.

    Defaults: clip_count=2, speaking_clip_index=1 (default), script_blocks[1] is
    the only non-silent entry.
    """
    defaults: dict = {
        "video_id": str(uuid.uuid4()),
        "product_id": "a3f9c12e7b04",
        "spec_index": 0,
        "tone": "warm storyteller",
        "narrative_arc": "A quiet morning with a ceramic coffee cup with branded packaging.",
        "talent_id": "talent_01",
        "clip_count": 2,
        "scene_descriptions": [
            "Talent picks up a ceramic coffee cup with branded packaging from the counter.",
            "Talent holds the cup at eye level; steam rises into morning light.",
        ],
        "script_blocks": ["", "Ciao mondo"],
        "speaking_clip_index": 1,
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return VideoSpec(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_videospec_happy_path_passes_validation() -> None:
    """clip_count=2, two scenes, script_blocks=["", "Ciao mondo"], speaking_clip_index=1 → valid."""
    spec = _make_spec(
        clip_count=2,
        scene_descriptions=[
            "Talent lifts a ceramic coffee cup with branded packaging.",
            "Talent sets the cup on a sunlit wooden surface.",
        ],
        script_blocks=["", "Ciao mondo"],
        speaking_clip_index=1,
    )
    assert spec.clip_count == 2
    assert spec.speaking_clip_index == 1
    assert spec.script_blocks[1] == "Ciao mondo"


def test_videospec_default_speaking_clip_index_is_1() -> None:
    """When speaking_clip_index is not supplied it defaults to 1."""
    spec = VideoSpec(
        video_id=str(uuid.uuid4()),
        product_id="a3f9c12e7b04",
        spec_index=0,
        tone="warm storyteller",
        narrative_arc="Morning ritual.",
        talent_id="talent_01",
        clip_count=2,
        scene_descriptions=["Scene A.", "Scene B."],
        script_blocks=["", "Buongiorno."],
    )
    assert spec.speaking_clip_index == 1


def test_videospec_clip_count_mismatch_scenes_raises() -> None:
    """3 clips but only 2 scene_descriptions → ValidationError."""
    with pytest.raises(ValidationError, match="scene_descriptions"):
        _make_spec(
            clip_count=3,
            scene_descriptions=["Scene A.", "Scene B."],  # only 2
            script_blocks=["", "Buongiorno.", ""],
        )


def test_videospec_clip_count_mismatch_scripts_raises() -> None:
    """2 clips but 3 script_blocks → ValidationError."""
    with pytest.raises(ValidationError, match="script_blocks"):
        _make_spec(
            clip_count=2,
            scene_descriptions=["Scene A.", "Scene B."],
            script_blocks=["", "Buongiorno.", ""],  # 3 entries
        )


def test_videospec_speaking_index_out_of_range_raises() -> None:
    """speaking_clip_index=5 with clip_count=2 → ValidationError."""
    with pytest.raises(ValidationError, match="speaking_clip_index"):
        _make_spec(
            clip_count=2,
            scene_descriptions=["Scene A.", "Scene B."],
            script_blocks=["", "Buongiorno."],
            speaking_clip_index=5,
        )


def test_videospec_speaking_index_negative_raises() -> None:
    """speaking_clip_index=-1 → ValidationError."""
    with pytest.raises(ValidationError, match="speaking_clip_index"):
        _make_spec(
            clip_count=2,
            scene_descriptions=["Scene A.", "Scene B."],
            script_blocks=["", "Buongiorno."],
            speaking_clip_index=-1,
        )


def test_videospec_no_speech_anywhere_raises() -> None:
    """All script_blocks empty → ValidationError (no non-silent block found)."""
    with pytest.raises(ValidationError, match="silent"):
        _make_spec(
            clip_count=2,
            scene_descriptions=["Scene A.", "Scene B."],
            script_blocks=["", ""],
        )


def test_videospec_speech_in_wrong_clip_raises() -> None:
    """speaking_clip_index=1 but speech is in clip 0 → ValidationError."""
    with pytest.raises(ValidationError, match="speaking_clip_index"):
        _make_spec(
            clip_count=2,
            scene_descriptions=["Scene A.", "Scene B."],
            script_blocks=["Spoken!", ""],  # speech at 0, not at 1
            speaking_clip_index=1,
        )


def test_videospec_multiple_speech_blocks_raises() -> None:
    """Two non-silent script_blocks with speaking_clip_index=1 → ValidationError."""
    with pytest.raises(ValidationError, match="Multiple"):
        _make_spec(
            clip_count=2,
            scene_descriptions=["Scene A.", "Scene B."],
            script_blocks=["Hello", "World"],  # both non-silent
            speaking_clip_index=1,
        )


def test_videospec_silent_marker_literal_silent_treated_as_silent() -> None:
    """'[silent]' is treated as a silent block; real speech at index 1 → valid."""
    spec = _make_spec(
        clip_count=2,
        scene_descriptions=["Scene A.", "Scene B."],
        script_blocks=["[silent]", "real speech"],
        speaking_clip_index=1,
    )
    assert spec.script_blocks[0] == "[silent]"
    assert spec.script_blocks[1] == "real speech"


def test_videospec_silent_marker_whitespace_only_treated_as_silent() -> None:
    """Whitespace-only string is treated as silent; real speech at index 1 → valid."""
    spec = _make_spec(
        clip_count=2,
        scene_descriptions=["Scene A.", "Scene B."],
        script_blocks=["   ", "real speech"],
        speaking_clip_index=1,
    )
    assert spec.script_blocks[0] == "   "
    assert spec.script_blocks[1] == "real speech"
