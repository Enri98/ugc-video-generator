"""Unit tests for ugc_pipeline.prompts.first_frame.

Covers VERSION, TEMPLATE, CONTENT_SHA256, render(), render_softened(),
render_with_reference(), and render_with_reference_softened().
"""

from __future__ import annotations

import hashlib
import re

import pytest

from ugc_pipeline.models import ProductBrief, VideoSpec
from ugc_pipeline.prompts import first_frame
from ugc_pipeline.prompts.first_frame import (
    CONTENT_SHA256,
    REFERENCE_PREAMBLE,
    TEMPLATE,
    VERSION,
    _SAFETY_PREAMBLE,
    render,
    render_softened,
    render_with_reference,
    render_with_reference_softened,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_brief(**overrides: object) -> ProductBrief:
    defaults: dict = {
        "product_id": "a3f9c12e7b04",
        "image_path": "artifacts/a3f9c12e7b04/product.jpg",
        "shape": "cylindrical mug with a C-shaped handle",
        "dominant_colours": ["#F5F0E8", "#3B2A1A"],
        "packaging_style": "kraft paper box with embossed geometric pattern and ribbon closure",
        "inferred_category": "kitchenware / home lifestyle",
        "lifestyle_contexts": [
            "morning routine at a kitchen counter",
            "desk setup during remote work",
            "outdoor picnic on a blanket",
        ],
        "visual_notes": "matte ceramic surface with slight speckle texture",
    }
    defaults.update(overrides)
    return ProductBrief(**defaults)  # type: ignore[arg-type]


def _make_spec(**overrides: object) -> VideoSpec:
    defaults: dict = {
        "video_id": "vid_test_001",
        "product_id": "a3f9c12e7b04",
        "spec_index": 0,
        "tone": "warm storyteller",
        "narrative_arc": "A slow-morning ritual with a favourite mug.",
        "talent_id": "talent_02",
        "clip_count": 3,
        "scene_descriptions": [
            "Talent holds the mug with both hands at a sunlit kitchen counter, steam rising.",
            "Close framing of talent's face, eyes closed, inhaling the aroma.",
            "Talent sets the mug on the counter and smiles at camera.",
        ],
        "script_blocks": [
            "[silent]",
            "Sento il profumo del caffè.",
            "[silent]",
        ],
        "speaking_clip_index": 1,
        "visual_style_notes": "Warm golden hour light, film grain, shallow depth of field.",
    }
    defaults.update(overrides)
    return VideoSpec(**defaults)  # type: ignore[arg-type]


_TALENT_DESCRIPTOR = "woman, late 20s, natural relaxed aesthetic, warm camera presence"


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_version_present(self) -> None:
        assert VERSION == "1.2.0"

    def test_template_non_empty(self) -> None:
        assert TEMPLATE.strip()

    def test_content_sha256_matches_template(self) -> None:
        expected = hashlib.sha256(TEMPLATE.encode()).hexdigest()
        assert CONTENT_SHA256 == expected

    def test_reference_preamble_non_empty(self) -> None:
        assert REFERENCE_PREAMBLE.strip()

    def test_reference_preamble_mentions_identity(self) -> None:
        # Must convey identity-preservation instruction.
        assert "identity" in REFERENCE_PREAMBLE.lower() or "preserved" in REFERENCE_PREAMBLE.lower()


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


class TestRender:
    def _rendered(self) -> str:
        return render(_make_spec(), _make_brief(), clip_index=0, talent_descriptor=_TALENT_DESCRIPTOR)

    def test_render_returns_baseline_prompt(self) -> None:
        result = self._rendered()
        # Scene text from clip 0
        assert "sunlit kitchen counter" in result
        # Talent descriptor injected
        assert _TALENT_DESCRIPTOR in result
        # Brand-neutral subject text derived from brief.shape
        assert "cylindrical mug" in result

    def test_no_unreplaced_placeholders_in_render(self) -> None:
        result = self._rendered()
        leftover = re.findall(r"\{[a-z_]+\}", result)
        assert leftover == [], f"Unreplaced placeholders found: {leftover}"

    def test_different_clips_produce_different_output(self) -> None:
        spec = _make_spec()
        brief = _make_brief()
        r0 = render(spec, brief, clip_index=0, talent_descriptor=_TALENT_DESCRIPTOR)
        r1 = render(spec, brief, clip_index=1, talent_descriptor=_TALENT_DESCRIPTOR)
        assert r0 != r1


# ---------------------------------------------------------------------------
# render_softened()
# ---------------------------------------------------------------------------


class TestRenderSoftened:
    def _rendered(self) -> str:
        return render_softened(
            _make_spec(), _make_brief(), clip_index=0, talent_descriptor=_TALENT_DESCRIPTOR
        )

    def test_render_softened_starts_with_safety_preamble(self) -> None:
        result = self._rendered()
        assert result.startswith(_SAFETY_PREAMBLE)

    def test_render_softened_differs_from_render(self) -> None:
        base = render(_make_spec(), _make_brief(), clip_index=0, talent_descriptor=_TALENT_DESCRIPTOR)
        softened = self._rendered()
        assert hashlib.sha256(base.encode()).hexdigest() != hashlib.sha256(softened.encode()).hexdigest()


# ---------------------------------------------------------------------------
# render_with_reference()
# ---------------------------------------------------------------------------


class TestRenderWithReference:
    def _rendered(self, clip_index: int = 1) -> str:
        return render_with_reference(
            _make_spec(), _make_brief(), clip_index=clip_index, talent_descriptor=_TALENT_DESCRIPTOR
        )

    def _base(self, clip_index: int = 1) -> str:
        return render(
            _make_spec(), _make_brief(), clip_index=clip_index, talent_descriptor=_TALENT_DESCRIPTOR
        )

    def test_render_with_reference_includes_reference_preamble(self) -> None:
        result = self._rendered()
        # At minimum the key identifying phrase must appear.
        assert "REFERENCE IMAGE PROVIDED" in result

    def test_render_with_reference_preamble_substring_present(self) -> None:
        result = self._rendered()
        # The constant itself must be a substring of the output.
        assert REFERENCE_PREAMBLE in result

    def test_render_with_reference_contains_same_body_text(self) -> None:
        result = self._rendered(clip_index=1)
        base = self._base(clip_index=1)
        # The base prompt body must appear inside the reference-augmented prompt.
        assert base in result

    def test_render_with_reference_differs_from_render(self) -> None:
        result = self._rendered()
        base = self._base()
        assert (
            hashlib.sha256(result.encode()).hexdigest()
            != hashlib.sha256(base.encode()).hexdigest()
        )

    def test_reference_preamble_appears_before_body(self) -> None:
        result = self._rendered()
        base = self._base()
        preamble_pos = result.index(REFERENCE_PREAMBLE)
        body_pos = result.index(base)
        assert preamble_pos < body_pos

    def test_no_unreplaced_placeholders_in_render_with_reference(self) -> None:
        result = self._rendered()
        leftover = re.findall(r"\{[a-z_]+\}", result)
        assert leftover == [], f"Unreplaced placeholders found: {leftover}"


# ---------------------------------------------------------------------------
# render_with_reference_softened()
# ---------------------------------------------------------------------------


class TestRenderWithReferenceSoftened:
    def _rendered(self, clip_index: int = 1) -> str:
        return render_with_reference_softened(
            _make_spec(), _make_brief(), clip_index=clip_index, talent_descriptor=_TALENT_DESCRIPTOR
        )

    def test_render_with_reference_softened_starts_with_safety_preamble(self) -> None:
        result = self._rendered()
        assert result.startswith(_SAFETY_PREAMBLE)

    def test_render_with_reference_softened_contains_reference_preamble(self) -> None:
        result = self._rendered()
        # Softening must not strip the reference preamble.
        assert REFERENCE_PREAMBLE in result

    def test_render_with_reference_softened_contains_body_text(self) -> None:
        result = self._rendered(clip_index=1)
        # Core scene text must survive softening.
        assert "cylindrical mug" in result
        assert _TALENT_DESCRIPTOR in result

    def test_no_unreplaced_placeholders_in_render_with_reference_softened(self) -> None:
        result = self._rendered()
        leftover = re.findall(r"\{[a-z_]+\}", result)
        assert leftover == [], f"Unreplaced placeholders found: {leftover}"

    def test_render_with_reference_softened_preamble_order(self) -> None:
        """Safety preamble must appear before reference instruction in output."""
        result = self._rendered()
        safety_pos = result.index(_SAFETY_PREAMBLE)
        reference_pos = result.index(REFERENCE_PREAMBLE)
        assert safety_pos < reference_pos, (
            "Expected _SAFETY_PREAMBLE to appear before REFERENCE_PREAMBLE, "
            f"but got positions {safety_pos} and {reference_pos}"
        )

    def test_render_with_reference_softened_preserves_reference_preamble_with_dummy_blocklist_word(
        self,
    ) -> None:
        """REFERENCE_PREAMBLE must survive softening even when blocklist words appear inside it.

        This is the key regression test for Bug #2: _soften() must only strip
        blocklist words from the prompt *body*, never from REFERENCE_PREAMBLE.
        The injected blocklist contains words that appear verbatim in REFERENCE_PREAMBLE
        ("REFERENCE", "image", "person"), so the old code (which applied stripping to
        the entire concatenated string) would corrupt REFERENCE_PREAMBLE and this test
        would fail. After the fix, REFERENCE_PREAMBLE must appear verbatim in the result.
        """
        from unittest.mock import patch

        # Inject blocklist words that DO appear inside REFERENCE_PREAMBLE.
        # "REFERENCE" is the first word of REFERENCE_PREAMBLE; "image" and "person"
        # also appear inside it. The pre-fix code would strip these, corrupting the preamble.
        adversarial_blocklist = ["REFERENCE", "image", "person"]

        with patch.object(first_frame, "_SAFETY_BLOCKLIST", new=adversarial_blocklist):
            result = render_with_reference_softened(
                _make_spec(), _make_brief(), clip_index=1, talent_descriptor=_TALENT_DESCRIPTOR
            )

        # The full REFERENCE_PREAMBLE constant must appear verbatim in the output.
        assert REFERENCE_PREAMBLE in result, (
            "REFERENCE_PREAMBLE was corrupted by blocklist stripping. "
            "The softening must only operate on the body, not the preamble."
        )
