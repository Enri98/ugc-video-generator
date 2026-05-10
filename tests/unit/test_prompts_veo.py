"""Unit tests for src/ugc_pipeline/prompts/veo.py."""

from __future__ import annotations

import hashlib
import re

import pytest

from ugc_pipeline.prompts import veo as veo_prompt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")

_SCENE = "Medium close-up: talent lifts the product from a sunlit kitchen counter."
_TONE = "warm storyteller"
_SCRIPT = "Questo cambia davvero la mia giornata."


# ---------------------------------------------------------------------------
# Module-level constant tests
# ---------------------------------------------------------------------------


def test_version_present() -> None:
    assert veo_prompt.VERSION == "1.0.0"


def test_template_non_empty() -> None:
    assert veo_prompt.TEMPLATE.strip()


def test_content_sha256_matches_template() -> None:
    expected = hashlib.sha256(veo_prompt.TEMPLATE.encode()).hexdigest()
    assert veo_prompt.CONTENT_SHA256 == expected


# ---------------------------------------------------------------------------
# render() — speaking clip
# ---------------------------------------------------------------------------


def test_render_speaking_clip_contains_scene_and_script() -> None:
    output = veo_prompt.render(
        scene_description=_SCENE,
        tone=_TONE,
        is_speaking_clip=True,
        script_block=_SCRIPT,
    )
    assert _SCENE in output
    assert _SCRIPT in output
    assert _TONE in output
    assert "AUDIO:" in output


def test_render_speaking_clip_with_empty_script_raises_value_error() -> None:
    with pytest.raises(ValueError):
        veo_prompt.render(
            scene_description=_SCENE,
            tone=_TONE,
            is_speaking_clip=True,
            script_block="",
        )


def test_render_speaking_clip_with_whitespace_script_raises_value_error() -> None:
    with pytest.raises(ValueError):
        veo_prompt.render(
            scene_description=_SCENE,
            tone=_TONE,
            is_speaking_clip=True,
            script_block="   ",
        )


# ---------------------------------------------------------------------------
# render() — silent clip
# ---------------------------------------------------------------------------


def test_render_silent_clip_contains_no_speech_marker() -> None:
    output = veo_prompt.render(
        scene_description=_SCENE,
        tone=_TONE,
        is_speaking_clip=False,
    )
    assert "AUDIO:" in output
    # Should signal absence of voice
    assert any(
        marker in output.lower()
        for marker in ("no spoken", "no voice", "ambient", "no music")
    )


def test_render_silent_clip_ignores_script_block() -> None:
    output = veo_prompt.render(
        scene_description=_SCENE,
        tone=_TONE,
        is_speaking_clip=False,
        script_block=_SCRIPT,
    )
    assert _SCRIPT not in output


# ---------------------------------------------------------------------------
# render() — no unreplaced placeholders
# ---------------------------------------------------------------------------


def test_render_no_unreplaced_placeholders_speaking() -> None:
    output = veo_prompt.render(
        scene_description=_SCENE,
        tone=_TONE,
        is_speaking_clip=True,
        script_block=_SCRIPT,
    )
    assert not _PLACEHOLDER_RE.search(output), (
        f"Unreplaced placeholder found in speaking output: {output!r}"
    )


def test_render_no_unreplaced_placeholders_silent() -> None:
    output = veo_prompt.render(
        scene_description=_SCENE,
        tone=_TONE,
        is_speaking_clip=False,
    )
    assert not _PLACEHOLDER_RE.search(output), (
        f"Unreplaced placeholder found in silent output: {output!r}"
    )


# Combine both checks under the single test name the spec requested
def test_render_no_unreplaced_placeholders() -> None:
    for is_speaking, block in [(True, _SCRIPT), (False, "")]:
        output = veo_prompt.render(
            scene_description=_SCENE,
            tone=_TONE,
            is_speaking_clip=is_speaking,
            script_block=block,
        )
        assert not _PLACEHOLDER_RE.search(output)


# ---------------------------------------------------------------------------
# render() — speaking vs silent differ
# ---------------------------------------------------------------------------


def test_speaking_and_silent_outputs_differ() -> None:
    speaking = veo_prompt.render(
        scene_description=_SCENE,
        tone=_TONE,
        is_speaking_clip=True,
        script_block=_SCRIPT,
    )
    silent = veo_prompt.render(
        scene_description=_SCENE,
        tone=_TONE,
        is_speaking_clip=False,
    )
    assert speaking != silent
