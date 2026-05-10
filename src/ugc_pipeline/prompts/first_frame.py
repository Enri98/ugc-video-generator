"""First-frame composite prompt template for the UGC pipeline.

See SPEC.md §6.3 for the canonical template shape and the coffee-cup fixture
rendering. This prompt is sent to Nano Banana 2 (Gemini 3 Flash Image) to
produce a 720x1280 PNG for each clip (9:16 vertical, matches the 720p Veo
output resolution configured in pipeline_config.yaml). No Italian content —
this step produces a visual asset.
"""

from __future__ import annotations

import hashlib

from ugc_pipeline.models import ProductBrief, VideoSpec

VERSION = "1.2.0"

REFERENCE_PREAMBLE = (
    "REFERENCE IMAGE PROVIDED: A reference image of the same talent and the same "
    "product is attached as the first input. Use it as the authoritative source for: "
    "the person's face, hair, body type, wardrobe, and the product's shape/colour/"
    "packaging. Only the scene, pose, framing, and lighting may change between this "
    "image and the reference — the identity of the person and product MUST be "
    "preserved exactly."
)

TEMPLATE = """\
Generate a photorealistic vertical portrait-format still image.
ASPECT RATIO: 9:16 vertical (portrait orientation, taller than wide). This is a TikTok / Reels / Shorts framing — strictly NOT square, NOT landscape.

SUBJECT: {subject_description}

PRODUCT SCALE: {product_size_hint}
The product MUST be rendered at realistic, hand-held human scale. Use the talent's hand and body as the size reference. Never render the product larger than the talent's torso. If the product appears oversized relative to the talent, the image is wrong.

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


_DEFAULT_SIZE_HINT = (
    "A small consumer product roughly 20-25 cm tall, sized to be held comfortably "
    "in one or two hands like a paperback book or a tall coffee mug."
)


def render(
    spec: VideoSpec,
    brief: ProductBrief,
    clip_index: int,
    talent_descriptor: str,
    product_size_hint: str = _DEFAULT_SIZE_HINT,
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

    # Talent action derived from scene description.
    # Pronoun-neutral phrasing — talent gender varies across the talent pool.
    talent_action = "Expression: calm, intentional. Looking away from the camera in this frame."

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
        product_size_hint=product_size_hint,
    )


_SAFETY_PREAMBLE = (
    "A tasteful, brand-safe still image with no people in close physical contact: "
)


def _soften(body: str, *, preamble: str = "") -> str:
    """Apply heuristic safety softening to an already-rendered prompt body.

    Strips blocklist words from *body* only, then returns:
        ``_SAFETY_PREAMBLE + preamble + body_softened``

    The *preamble* argument (e.g. ``REFERENCE_PREAMBLE + "\\n\\n"``) is
    inserted verbatim between the safety preamble and the softened body —
    blocklist replacement is deliberately NOT applied to it. Rule-based only —
    no LLM involvement.
    """
    softened = body
    for word in _SAFETY_BLOCKLIST:
        softened = softened.replace(word, "")
        softened = softened.replace(word.capitalize(), "")
    return _SAFETY_PREAMBLE + preamble + softened


def render_softened(
    spec: VideoSpec,
    brief: ProductBrief,
    clip_index: int,
    talent_descriptor: str,
    product_size_hint: str = _DEFAULT_SIZE_HINT,
) -> str:
    """Return a softened variant of the first-frame prompt.

    This is a heuristic approach: it prepends a brand-safe preamble and strips
    words from a small blocklist that are most likely to trigger safety filters.
    No LLM is involved in the softening — it is entirely rule-based.

    Note: This is intentionally conservative and may produce slightly awkward
    phrasing. The goal is to pass safety filters on a single retry, not to
    produce the ideal prompt.
    """
    base_prompt = render(spec, brief, clip_index, talent_descriptor, product_size_hint)
    return _soften(base_prompt)


def render_with_reference(
    spec: VideoSpec,
    brief: ProductBrief,
    clip_index: int,
    talent_descriptor: str,
    product_size_hint: str = _DEFAULT_SIZE_HINT,
) -> str:
    """Render the first-frame prompt for clip_index, prepending the reference-image preamble.

    For clip i ≥ 1, the caller should attach the clip-0 PNG bytes as the first
    image input alongside this prompt. The preamble instructs the model to anchor
    person identity and product appearance on the reference image; only scene,
    pose, framing, and lighting may vary.

    For clip 0 the caller should still use the plain ``render()`` function (no
    reference image exists yet). This function does not enforce that constraint
    in code — the caller is responsible for the clip-0 / clip-i distinction.

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
    product_size_hint:
        Optional override for the product scale description.

    Returns
    -------
    str
        The fully rendered image-generation prompt with the REFERENCE_PREAMBLE
        prepended.
    """
    base_prompt = render(spec, brief, clip_index, talent_descriptor, product_size_hint)
    return REFERENCE_PREAMBLE + "\n\n" + base_prompt


def render_with_reference_softened(
    spec: VideoSpec,
    brief: ProductBrief,
    clip_index: int,
    talent_descriptor: str,
    product_size_hint: str = _DEFAULT_SIZE_HINT,
) -> str:
    """Return a softened, reference-anchored variant of the first-frame prompt.

    Combines ``render_with_reference()`` with the heuristic safety softening
    applied by ``render_softened()``. The output order is: safety preamble
    first, then the reference instruction, then the base prompt body.

    The reference instruction (``REFERENCE_PREAMBLE``) is left intact — only
    the body of the rendered prompt is softened (blocklist words stripped).
    This guarantees that identity-preservation instructions cannot be
    accidentally mangled by future blocklist changes.

    Parameters
    ----------
    spec:
        The VideoSpec for the current video.
    brief:
        The ProductBrief describing the product.
    clip_index:
        Which clip (0-based) this first frame is for.
    talent_descriptor:
        Human-readable description of the talent resolved from the talent pool.
    product_size_hint:
        Optional override for the product scale description.

    Returns
    -------
    str
        The fully rendered, softened, reference-anchored prompt.
    """
    base_prompt = render(spec, brief, clip_index, talent_descriptor, product_size_hint)
    return _soften(base_prompt, preamble=REFERENCE_PREAMBLE + "\n\n")
