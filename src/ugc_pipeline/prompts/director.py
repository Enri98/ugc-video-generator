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

VERSION = "1.0.0"

TONES: list[str] = [
    "warm storyteller",
    "energetic lifestyle",
    "serene ASMR",
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

REQUIREMENTS:
- clip_count: {clip_count}
- narrative_arc: one sentence describing the emotional journey of the video
- scene_descriptions: English, one per clip, vivid and cinematographic
- script_blocks: Italian voiceover lines, one per clip  # Italian content below
- visual_style_notes: lighting, colour grading, camera motion suggestions

Respond in valid JSON conforming to the VideoSpec schema.\
"""

CONTENT_SHA256 = hashlib.sha256(TEMPLATE.encode()).hexdigest()


def render(
    brief: ProductBrief,
    talent_id: str,
    talent_descriptor: str,
    tone: str,
    other_tones: list[str],
    clip_count: int,
    lifestyle_context: str,
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
    """
    other_tones_str = " and ".join(f'"{t}"' for t in other_tones)
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
    )
