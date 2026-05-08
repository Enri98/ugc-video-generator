"""First-frame composite prompt template for the UGC pipeline.

See SPEC.md §6.3 for the canonical template shape and the coffee-cup fixture
rendering. This prompt is sent to Nano Banana 2 (Gemini 3 Flash Image) to
produce a 1080x1920 PNG for each clip. No Italian content — this step produces
a visual asset.
"""

from __future__ import annotations

import hashlib

from ugc_pipeline.models import ProductBrief, VideoSpec

VERSION = "1.0.0"

TEMPLATE = """\
Generate a photorealistic vertical portrait-format still image (1080x1920 px, 9:16 aspect ratio).

SUBJECT: {subject_description}

TALENT: {talent_descriptor}
{talent_action}

SCENE: {scene_setting}
{colour_palette}

STYLE: {visual_style_notes}
No text, watermarks, logos, or brand identifiers in the image.

This image will be used as the first frame of a video clip. {forward_motion_hint}\
"""

CONTENT_SHA256 = hashlib.sha256(TEMPLATE.encode()).hexdigest()

# Words that may trigger safety filters; used by render_softened (heuristic only).
_SAFETY_BLOCKLIST: list[str] = ["close-up", "intimate", "embrace"]


def render(
    spec: VideoSpec,
    brief: ProductBrief,
    clip_index: int,
    talent_descriptor: str,
) -> str:
    """Render the first-frame prompt for *clip_index* in *spec*.

    Parameters
    ----------
    spec:
        The VideoSpec for the current video.
    brief:
        The ProductBrief describing the product.
    clip_index:
        Which clip (0-based) this first frame is for.
    talent_descriptor:
        Human-readable description of the talent (gender, age, aesthetic, etc.)
        resolved from the talent pool.

    Returns
    -------
    str
        The fully rendered image-generation prompt.
    """
    scene_description = spec.scene_descriptions[clip_index]

    # Build subject description from brief attributes
    subject_description = (
        f"A {brief.shape}, described as: {brief.packaging_style}."
    )

    # Build scene setting from the spec's scene description
    colour_palette_str = ", ".join(brief.dominant_colours) if brief.dominant_colours else "neutral tones"
    scene_setting = scene_description

    colour_palette = f"Colour palette: {colour_palette_str}. Shallow depth of field; background softly blurred."

    # Visual style notes from spec, or a sensible default
    visual_style_notes = (
        spec.visual_style_notes
        if spec.visual_style_notes
        else "Photorealistic UGC aesthetic. Natural skin tones. No studio lighting artefacts."
    )

    # Talent action derived from scene description
    talent_action = "Expression: calm, intentional. She is not looking at the camera in this frame."

    # Forward motion hint is a fixed string per SPEC.md §5 step 4
    forward_motion_hint = (
        "Ensure composition allows natural forward motion into the next clip."
    )

    return TEMPLATE.format(
        subject_description=subject_description,
        talent_descriptor=talent_descriptor,
        talent_action=talent_action,
        scene_setting=scene_setting,
        colour_palette=colour_palette,
        visual_style_notes=visual_style_notes,
        forward_motion_hint=forward_motion_hint,
    )


def render_softened(
    spec: VideoSpec,
    brief: ProductBrief,
    clip_index: int,
    talent_descriptor: str,
) -> str:
    """Return a softened variant of the first-frame prompt.

    This is a heuristic approach: it prepends a brand-safe preamble and strips
    words from a small blocklist that are most likely to trigger safety filters.
    No LLM is involved in the softening — it is entirely rule-based.

    Note: This is intentionally conservative and may produce slightly awkward
    phrasing. The goal is to pass safety filters on a single retry, not to
    produce the ideal prompt.
    """
    base_prompt = render(spec, brief, clip_index, talent_descriptor)

    # Strip blocklist words (case-insensitive, whole-word-ish replacement)
    softened = base_prompt
    for word in _SAFETY_BLOCKLIST:
        softened = softened.replace(word, "")
        softened = softened.replace(word.capitalize(), "")

    # Prepend the safety preamble
    preamble = (
        "A tasteful, brand-safe still image with no people in close physical contact: "
    )
    return preamble + softened
