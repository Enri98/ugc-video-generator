"""Unit tests for prompt template modules.

Covers analyst.py and director.py — both in ugc_pipeline.prompts.
"""

from __future__ import annotations

import hashlib

import pytest

from ugc_pipeline.models import ProductBrief
from ugc_pipeline.prompts import analyst, director
from ugc_pipeline.prompts.director import TONES


# ---------------------------------------------------------------------------
# Shared fixtures
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


_TALENT_POOL: dict = {
    "talent_02": {
        "gender": "woman",
        "age_range": "late 20s",
        "aesthetic": "natural, relaxed, warm",
        "camera_relationship": "conversational",
        "lighting_preference": "soft natural window light",
    }
}


# ---------------------------------------------------------------------------
# Analyst module
# ---------------------------------------------------------------------------


class TestAnalystModule:
    def test_version_present(self) -> None:
        assert analyst.VERSION == "1.0.0"

    def test_template_non_empty(self) -> None:
        assert analyst.TEMPLATE.strip()

    def test_content_sha256_matches_template(self) -> None:
        expected = hashlib.sha256(analyst.TEMPLATE.encode()).hexdigest()
        assert analyst.CONTENT_SHA256 == expected

    def test_render_returns_template_verbatim(self) -> None:
        result = analyst.render()
        assert result == analyst.TEMPLATE

    def test_render_idempotent(self) -> None:
        assert analyst.render() == analyst.render()


# ---------------------------------------------------------------------------
# Director module — module-level constants
# ---------------------------------------------------------------------------


class TestDirectorModuleConstants:
    def test_version_present(self) -> None:
        assert director.VERSION == "1.1.0"

    def test_template_non_empty(self) -> None:
        assert director.TEMPLATE.strip()

    def test_content_sha256_matches_template(self) -> None:
        expected = hashlib.sha256(director.TEMPLATE.encode()).hexdigest()
        assert director.CONTENT_SHA256 == expected

    def test_tones_has_exactly_three_entries(self) -> None:
        assert len(TONES) == 3

    def test_tones_are_distinct(self) -> None:
        assert len(set(TONES)) == 3

    def test_canonical_tone_names(self) -> None:
        assert "warm storyteller" in TONES
        assert "energetic lifestyle" in TONES
        assert "serene ASMR" in TONES


# ---------------------------------------------------------------------------
# Director render — substitution correctness
# ---------------------------------------------------------------------------


class TestDirectorRender:
    def _render_spec0(self) -> str:
        brief = _make_brief()
        from ugc_pipeline.utils.config import get_talent_descriptor

        tone = TONES[0]
        other_tones = [t for t in TONES if t != tone]
        return director.render(
            brief=brief,
            talent_id="talent_02",
            talent_descriptor=get_talent_descriptor("talent_02", _TALENT_POOL),
            tone=tone,
            other_tones=other_tones,
            clip_count=2,
            lifestyle_context=brief.lifestyle_contexts[0],
        )

    def test_render_contains_chosen_tone(self) -> None:
        result = self._render_spec0()
        assert "warm storyteller" in result

    def test_render_does_not_contain_other_tones_in_tone_field(self) -> None:
        result = self._render_spec0()
        # The chosen tone line should name the tone; other tones appear only in the
        # "do NOT use" clause — verify the chosen tone is the one highlighted
        assert "TONE FOR THIS SPEC: warm storyteller" in result

    def test_render_references_both_other_tones(self) -> None:
        result = self._render_spec0()
        assert "energetic lifestyle" in result
        assert "serene ASMR" in result

    def test_render_contains_talent_descriptor(self) -> None:
        from ugc_pipeline.utils.config import get_talent_descriptor

        desc = get_talent_descriptor("talent_02", _TALENT_POOL)
        result = self._render_spec0()
        assert desc in result

    def test_render_contains_shape(self) -> None:
        result = self._render_spec0()
        assert "cylindrical mug" in result

    def test_render_contains_clip_count(self) -> None:
        result = self._render_spec0()
        assert "clip_count: 2" in result

    def test_render_contains_lifestyle_context(self) -> None:
        result = self._render_spec0()
        assert "morning routine at a kitchen counter" in result

    def test_different_spec_indices_produce_different_output(self) -> None:
        brief = _make_brief()
        from ugc_pipeline.utils.config import get_talent_descriptor

        rendered = []
        for spec_index in range(3):
            tone = TONES[spec_index]
            other_tones = [t for t in TONES if t != tone]
            rendered.append(
                director.render(
                    brief=brief,
                    talent_id="talent_02",
                    talent_descriptor=get_talent_descriptor("talent_02", _TALENT_POOL),
                    tone=tone,
                    other_tones=other_tones,
                    clip_count=2,
                    lifestyle_context=brief.lifestyle_contexts[spec_index % len(brief.lifestyle_contexts)],
                )
            )

        sha = [hashlib.sha256(r.encode()).hexdigest() for r in rendered]
        assert sha[0] != sha[1]
        assert sha[1] != sha[2]
        assert sha[0] != sha[2]

    def test_no_unreplaced_placeholders(self) -> None:
        """The rendered string should not contain raw curly-brace placeholders."""
        result = self._render_spec0()
        import re

        # Match {word} patterns — legitimate ones should all be substituted
        leftover = re.findall(r"\{[a-z_]+\}", result)
        assert leftover == [], f"Unreplaced placeholders found: {leftover}"

    def test_brand_guidance_absent_by_default(self) -> None:
        """Without brand_guidance, no BRAND GUIDANCE section appears."""
        result = self._render_spec0()
        assert "BRAND GUIDANCE" not in result

    def test_brand_guidance_section_rendered_when_supplied(self) -> None:
        """Supplying brand_guidance injects positioning, must_avoid, should_do."""
        from ugc_pipeline.utils.config import get_talent_descriptor

        brief = _make_brief()
        guidance = {
            "positioning": "Calm, conversational introduction of a closed product.",
            "must_avoid": ["Do NOT unbox the product."],
            "should_do": ["Hold the closed package and speak about it."],
        }
        result = director.render(
            brief=brief,
            talent_id="talent_02",
            talent_descriptor=get_talent_descriptor("talent_02", _TALENT_POOL),
            tone=TONES[0],
            other_tones=[t for t in TONES if t != TONES[0]],
            clip_count=2,
            lifestyle_context=brief.lifestyle_contexts[0],
            brand_guidance=guidance,
        )
        assert "BRAND GUIDANCE" in result
        assert "Calm, conversational introduction" in result
        assert "Do NOT unbox the product." in result
        assert "Hold the closed package and speak about it." in result
