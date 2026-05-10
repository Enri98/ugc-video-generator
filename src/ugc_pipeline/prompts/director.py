"""Creative director prompt template for the UGC pipeline.

See SPEC.md §6.2 for the canonical template shape and the coffee-cup fixture
rendering. Each call produces one VideoSpec; the orchestrator calls this three
times with different tones.

TONES: the three canonical tones for the pipeline. The orchestrator assigns
one tone per spec_index and passes the other two via other_tones so the model
is explicitly instructed to avoid them.
"""

from __future__ import annotations

import hashlib

from ugc_pipeline.models import ProductBrief

VERSION = "1.4.0"

# Three rotating tones for the three spec_indices per product. Each tone must be
# compatible with the single-speaking-clip pattern (one short spoken line, the rest
# silent ambient). ASMR removed in v1.3 because its continuous-audio nature
# conflicted with consolidation.
TONES: list[str] = [
    "warm storyteller",
    "energetic lifestyle",
    "intimate confidant",
]

TEMPLATE = """\
You are a creative director for a UGC video campaign for a consumer brand.
You will generate one VideoSpec for a short-form vertical video (15–30 seconds).

PRODUCT BRIEF:
Shape: {shape}
Category: {category}
Lifestyle context: {lifestyle_context}
Packaging style: {packaging_style}
Visual notes: {visual_notes}

TALENT:
talent_id: {talent_id}
Descriptor: {talent_descriptor}

TONE FOR THIS SPEC: {tone}
(The other two specs in this batch will use "{other_tones}" — do NOT use those tones here.)
{brand_guidance_section}
VOICE CONSOLIDATION RULE (read carefully before writing script_blocks):
Voice consistency across independently generated video clips is not guaranteed —
each clip is rendered by a separate model call with no shared audio context.
To ensure a coherent viewer experience, ALL spoken dialogue is consolidated into
a single "speaking clip". Every other clip is visually expressive (allusive facial
expressions, holding or looking at the product) but contains NO spoken dialogue.

SPEAKING CLIP: clip index {speaking_clip_index} (0-based, out of {clip_count} total clips).
SILENT CLIPS:  all other indices — {silent_clip_indices}.

REQUIREMENTS:
- clip_count: {clip_count}
- speaking_clip_index: {speaking_clip_index}  # must appear in the JSON output unchanged
- narrative_arc: one sentence describing the emotional journey of the video
- scene_descriptions: English, one per clip, vivid and cinematographic
- script_blocks: exactly {clip_count} entries.
    * script_blocks[{speaking_clip_index}]: one calm, conversational Italian voiceover line,
      approximately 6–8 seconds when spoken aloud (fits in a single 8-second clip).
    * script_blocks[i] for all other i: the literal string "[silent]" — no dialogue.
- visual_style_notes: lighting, colour grading, camera motion suggestions

Respond in valid JSON conforming to the VideoSpec schema.\
"""

CONTENT_SHA256 = hashlib.sha256(TEMPLATE.encode()).hexdigest()


def _substitute_guidance_placeholders(
    text: str,
    *,
    brand_name: str,
    product_name: str,
    speaking_clip_index: int,
) -> str:
    """Substitute the three allowed placeholders in a guidance string.

    Placeholders that are not one of {brand_name}, {product_name},
    {speaking_clip_index} are left untouched (do NOT crash on stray '{').
    """
    return (
        text.replace("{brand_name}", brand_name)
            .replace("{product_name}", product_name)
            .replace("{speaking_clip_index}", str(speaking_clip_index))
    )


def _render_brand_guidance_section(
    guidance: dict | None,
    *,
    product_name: str = "",
    speaking_clip_index: int = 1,
) -> str:
    """Return a formatted BRAND GUIDANCE block, or '' when guidance is empty.

    Empty by default so behaviour and content_sha256 of existing prompts are
    preserved when the optional brand_guidance.yaml is not present.

    Strings in positioning, must_avoid, should_do, continuity, and
    naming_requirement may contain three substitutable placeholders:
    {brand_name}, {product_name}, {speaking_clip_index}.
    """
    if not guidance:
        return ""

    brand_name: str = guidance.get("brand_name", "") or ""

    def _sub(text: str) -> str:
        return _substitute_guidance_placeholders(
            text,
            brand_name=brand_name,
            product_name=product_name,
            speaking_clip_index=speaking_clip_index,
        )

    lines: list[str] = ["", "BRAND GUIDANCE (authoritative — overrides defaults):"]

    if brand_name:
        lines.append(f"Brand: {brand_name}")

    positioning = guidance.get("positioning")
    if positioning:
        lines.append(f"Positioning:\n{_sub(positioning.strip())}")

    must_avoid = guidance.get("must_avoid") or []
    if must_avoid:
        lines.append("MUST AVOID:")
        lines.extend(f"  - {_sub(rule)}" for rule in must_avoid)

    should_do = guidance.get("should_do") or []
    if should_do:
        lines.append("SHOULD DO:")
        lines.extend(f"  - {_sub(rule)}" for rule in should_do)

    continuity = guidance.get("continuity") or []
    if continuity:
        lines.append("CONTINUITY (apply across all clips of one video):")
        lines.extend(f"  - {_sub(rule)}" for rule in continuity)

    naming_requirement = guidance.get("naming_requirement")
    if naming_requirement:
        lines.append("NAMING REQUIREMENT:")
        lines.append(_sub(naming_requirement.strip()))

    return "\n".join(lines) + "\n"


def render(
    brief: ProductBrief,
    talent_id: str,
    talent_descriptor: str,
    tone: str,
    other_tones: list[str],
    clip_count: int,
    lifestyle_context: str,
    brand_guidance: dict | None = None,
    speaking_clip_index: int = 1,  # orchestrator should override this default
    *,
    product_name: str = "",
) -> str:
    """Render the director prompt by substituting all template placeholders.

    Parameters
    ----------
    brief:
        The ProductBrief for the product being directed.
    talent_id:
        The ID key of the chosen talent (e.g. ``"talent_02"``).
    talent_descriptor:
        A one-line English description of the talent from utils.config.get_talent_descriptor.
    tone:
        The narrative tone for this spec (one of TONES).
    other_tones:
        The remaining two tones that must NOT be used in this spec.
    clip_count:
        Number of clips (2 or 3) for this spec.
    lifestyle_context:
        One entry from brief.lifestyle_contexts chosen for this spec.
    brand_guidance:
        Optional brand guidance dict from brand_guidelines.yaml.
    speaking_clip_index:
        0-based index of the single clip that carries spoken dialogue.
        All other clips will be marked "[silent]". Defaults to 1.
    product_name:
        Human-readable product name derived from the input image filename.
        Used for placeholder substitution in brand_guidance strings.
    """
    other_tones_str = " and ".join(f'"{t}"' for t in other_tones)
    silent_clip_indices = ", ".join(
        str(i) for i in range(clip_count) if i != speaking_clip_index
    )
    return TEMPLATE.format(
        shape=brief.shape,
        category=brief.inferred_category,
        lifestyle_context=lifestyle_context,
        packaging_style=brief.packaging_style,
        visual_notes=brief.visual_notes or "not specified",
        talent_id=talent_id,
        talent_descriptor=talent_descriptor,
        tone=tone,
        other_tones=other_tones_str,
        clip_count=clip_count,
        brand_guidance_section=_render_brand_guidance_section(
            brand_guidance,
            product_name=product_name,
            speaking_clip_index=speaking_clip_index,
        ),
        speaking_clip_index=speaking_clip_index,
        silent_clip_indices=silent_clip_indices,
    )
