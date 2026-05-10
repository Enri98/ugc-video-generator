"""Unit tests for ugc_pipeline.prompts.first_frame.

Covers VERSION, TEMPLATE, CONTENT_SHA256, render(), render_softened(),
render_with_reference(), render_with_reference_softened(),
render_with_source(), render_with_source_softened(),
render_with_source_and_talent(), render_with_source_and_talent_softened().
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
    SOURCE_REFERENCE_PREAMBLE,
    TEMPLATE,
    VERSION,
    _SAFETY_PREAMBLE,
    render,
    render_softened,
    render_with_reference,
    render_with_reference_softened,
    render_with_source,
    render_with_source_and_talent,
    render_with_source_and_talent_softened,
    render_with_source_softened,
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
        assert VERSION == "1.3.1"

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

    def test_reference_preamble_does_not_claim_second_input(self) -> None:
        """REFERENCE_PREAMBLE must be position-neutral — no 'second input' claim.

        The preamble is used both when only one image is attached (degraded path)
        and when two images are attached (dual-reference path). Claiming 'second
        input' in the shared constant would mislead the model on the degraded path.
        """
        assert "second input" not in REFERENCE_PREAMBLE.lower(), (
            "REFERENCE_PREAMBLE must not claim a specific input position; "
            "that ordering hint belongs only in render_with_source_and_talent."
        )

    def test_source_reference_preamble_constant_present(self) -> None:
        """SOURCE_REFERENCE_PREAMBLE must exist and be non-empty."""
        assert SOURCE_REFERENCE_PREAMBLE.strip()

    def test_source_reference_preamble_mentions_product(self) -> None:
        """SOURCE_REFERENCE_PREAMBLE must mention 'product' (authoritative source instruction)."""
        assert "product" in SOURCE_REFERENCE_PREAMBLE.lower()

    def test_template_does_not_forbid_labels(self) -> None:
        """TEMPLATE must NOT contain 'No text, watermarks, logos' — that line was removed in v1.3.0."""
        assert "No text, watermarks, logos" not in TEMPLATE
        assert "brand identifiers" not in TEMPLATE


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
# render_with_reference()  (backward compat — talent-only)
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
        assert "TALENT REFERENCE IMAGE PROVIDED" in result

    def test_render_with_reference_preamble_substring_present(self) -> None:
        result = self._rendered()
        assert REFERENCE_PREAMBLE in result

    def test_render_with_reference_contains_same_body_text(self) -> None:
        result = self._rendered(clip_index=1)
        base = self._base(clip_index=1)
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
# render_with_reference_softened()  (backward compat — talent-only)
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
        ("TALENT", "image", "person"), so the old code (which applied stripping to
        the entire concatenated string) would corrupt REFERENCE_PREAMBLE and this test
        would fail. After the fix, REFERENCE_PREAMBLE must appear verbatim in the result.
        """
        from unittest.mock import patch

        # Inject blocklist words that DO appear inside REFERENCE_PREAMBLE.
        adversarial_blocklist = ["TALENT", "image", "person"]

        with patch.object(first_frame, "_SAFETY_BLOCKLIST", new=adversarial_blocklist):
            result = render_with_reference_softened(
                _make_spec(), _make_brief(), clip_index=1, talent_descriptor=_TALENT_DESCRIPTOR
            )

        # The full REFERENCE_PREAMBLE constant must appear verbatim in the output.
        assert REFERENCE_PREAMBLE in result, (
            "REFERENCE_PREAMBLE was corrupted by blocklist stripping. "
            "The softening must only operate on the body, not the preamble."
        )


# ---------------------------------------------------------------------------
# render_with_source() — clip 0: source product reference only
# ---------------------------------------------------------------------------


class TestRenderWithSource:
    def _rendered(self, clip_index: int = 0) -> str:
        return render_with_source(
            _make_spec(), _make_brief(), clip_index=clip_index, talent_descriptor=_TALENT_DESCRIPTOR
        )

    def test_render_with_source_includes_source_preamble(self) -> None:
        result = self._rendered()
        assert SOURCE_REFERENCE_PREAMBLE in result

    def test_render_with_source_does_not_include_talent_preamble(self) -> None:
        """Clip 0 should have source reference but NOT the talent reference preamble."""
        result = self._rendered()
        assert REFERENCE_PREAMBLE not in result

    def test_render_with_source_source_preamble_before_body(self) -> None:
        result = self._rendered()
        base = render(
            _make_spec(), _make_brief(), clip_index=0, talent_descriptor=_TALENT_DESCRIPTOR
        )
        src_pos = result.index(SOURCE_REFERENCE_PREAMBLE)
        body_pos = result.index(base)
        assert src_pos < body_pos

    def test_render_with_source_no_unreplaced_placeholders(self) -> None:
        result = self._rendered()
        leftover = re.findall(r"\{[a-z_]+\}", result)
        assert leftover == [], f"Unreplaced placeholders found: {leftover}"

    def test_render_with_source_contains_body_text(self) -> None:
        result = self._rendered()
        assert "cylindrical mug" in result
        assert _TALENT_DESCRIPTOR in result


# ---------------------------------------------------------------------------
# render_with_source_and_talent() — clips ≥1: both preambles
# ---------------------------------------------------------------------------


class TestRenderWithSourceAndTalent:
    def _rendered(self, clip_index: int = 1) -> str:
        return render_with_source_and_talent(
            _make_spec(), _make_brief(), clip_index=clip_index, talent_descriptor=_TALENT_DESCRIPTOR
        )

    def test_render_with_source_and_talent_includes_both_preambles(self) -> None:
        result = self._rendered()
        assert SOURCE_REFERENCE_PREAMBLE in result
        assert REFERENCE_PREAMBLE in result

    def test_render_with_source_and_talent_order(self) -> None:
        """SOURCE_REFERENCE_PREAMBLE must come BEFORE REFERENCE_PREAMBLE in output."""
        result = self._rendered()
        src_pos = result.index(SOURCE_REFERENCE_PREAMBLE)
        ref_pos = result.index(REFERENCE_PREAMBLE)
        assert src_pos < ref_pos, (
            f"Expected SOURCE_REFERENCE_PREAMBLE before REFERENCE_PREAMBLE, "
            f"got positions {src_pos} and {ref_pos}"
        )

    def test_render_with_source_and_talent_no_unreplaced_placeholders(self) -> None:
        result = self._rendered()
        leftover = re.findall(r"\{[a-z_]+\}", result)
        assert leftover == [], f"Unreplaced placeholders found: {leftover}"

    def test_render_with_source_and_talent_contains_body_text(self) -> None:
        result = self._rendered()
        assert "cylindrical mug" in result
        assert _TALENT_DESCRIPTOR in result

    def test_render_with_source_and_talent_contains_input_order_note(self) -> None:
        """Dual-reference output must include 'FIRST input' and 'SECOND input' ordering hints.

        These appear in _INPUT_ORDER_NOTE which is prepended only by
        render_with_source_and_talent — so the claim is accurate (two images ARE attached).
        """
        result = self._rendered()
        assert "FIRST input" in result, (
            "render_with_source_and_talent output must contain 'FIRST input' ordering hint"
        )
        assert "SECOND input" in result, (
            "render_with_source_and_talent output must contain 'SECOND input' ordering hint"
        )


# ---------------------------------------------------------------------------
# render_with_source_softened() — clip 0 safety retry
# ---------------------------------------------------------------------------


class TestRenderWithSourceSoftened:
    def _rendered(self, clip_index: int = 0) -> str:
        return render_with_source_softened(
            _make_spec(), _make_brief(), clip_index=clip_index, talent_descriptor=_TALENT_DESCRIPTOR
        )

    def test_render_with_source_softened_starts_with_safety_preamble(self) -> None:
        result = self._rendered()
        assert result.startswith(_SAFETY_PREAMBLE)

    def test_render_with_source_softened_preserves_source_preamble(self) -> None:
        result = self._rendered()
        assert SOURCE_REFERENCE_PREAMBLE in result

    def test_render_with_source_softened_does_not_include_talent_preamble(self) -> None:
        result = self._rendered()
        assert REFERENCE_PREAMBLE not in result

    def test_render_with_source_softened_order(self) -> None:
        """Safety preamble must appear before source preamble in output."""
        result = self._rendered()
        safety_pos = result.index(_SAFETY_PREAMBLE)
        src_pos = result.index(SOURCE_REFERENCE_PREAMBLE)
        assert safety_pos < src_pos

    def test_render_with_source_softened_no_unreplaced_placeholders(self) -> None:
        result = self._rendered()
        leftover = re.findall(r"\{[a-z_]+\}", result)
        assert leftover == [], f"Unreplaced placeholders found: {leftover}"


# ---------------------------------------------------------------------------
# render_with_source_and_talent_softened() — clip ≥1 safety retry
# ---------------------------------------------------------------------------


class TestRenderWithSourceAndTalentSoftened:
    def _rendered(self, clip_index: int = 1) -> str:
        return render_with_source_and_talent_softened(
            _make_spec(), _make_brief(), clip_index=clip_index, talent_descriptor=_TALENT_DESCRIPTOR
        )

    def test_render_with_source_and_talent_softened_starts_with_safety_preamble(self) -> None:
        result = self._rendered()
        assert result.startswith(_SAFETY_PREAMBLE)

    def test_render_with_source_and_talent_softened_preserves_both_preambles(self) -> None:
        """Softening must not strip SOURCE_REFERENCE_PREAMBLE or REFERENCE_PREAMBLE."""
        result = self._rendered()
        assert SOURCE_REFERENCE_PREAMBLE in result
        assert REFERENCE_PREAMBLE in result

    def test_render_with_source_and_talent_softened_order(self) -> None:
        """Order: safety → source reference → talent reference → body."""
        result = self._rendered()
        safety_pos = result.index(_SAFETY_PREAMBLE)
        src_pos = result.index(SOURCE_REFERENCE_PREAMBLE)
        ref_pos = result.index(REFERENCE_PREAMBLE)
        assert safety_pos < src_pos < ref_pos

    def test_render_with_source_and_talent_softened_no_unreplaced_placeholders(self) -> None:
        result = self._rendered()
        leftover = re.findall(r"\{[a-z_]+\}", result)
        assert leftover == [], f"Unreplaced placeholders found: {leftover}"

    def test_render_with_source_and_talent_softened_preserves_preambles_with_adversarial_blocklist(
        self,
    ) -> None:
        """Both preambles survive blocklist softening even when blocklist words appear in them."""
        from unittest.mock import patch

        adversarial_blocklist = ["SOURCE", "TALENT", "image", "product"]

        with patch.object(first_frame, "_SAFETY_BLOCKLIST", new=adversarial_blocklist):
            result = render_with_source_and_talent_softened(
                _make_spec(), _make_brief(), clip_index=1, talent_descriptor=_TALENT_DESCRIPTOR
            )

        assert SOURCE_REFERENCE_PREAMBLE in result, (
            "SOURCE_REFERENCE_PREAMBLE was corrupted by blocklist stripping."
        )
        assert REFERENCE_PREAMBLE in result, (
            "REFERENCE_PREAMBLE was corrupted by blocklist stripping."
        )
