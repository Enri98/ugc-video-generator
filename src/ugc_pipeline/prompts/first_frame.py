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

VERSION = "1.3.1"

SOURCE_REFERENCE_PREAMBLE = (
    "SOURCE PRODUCT IMAGE PROVIDED: A photo of the actual product is attached"
    " as the first input. This is the AUTHORITATIVE source for the product's"
    " appearance: shape, colour, packaging, and any printed text or labels"
    " visible on it. Reproduce these faithfully — the product on screen must"
    " be recognisable as the same item shown in the reference."
)

REFERENCE_PREAMBLE = (
    "TALENT REFERENCE IMAGE PROVIDED: An image of the same talent and the"
    " same product is attached as a reference input. Use it as the"
    " authoritative source for: the person's face, hair, body type, and"
    " wardrobe. Only the scene, pose, framing, and lighting may change"
    " between this image and the reference — the identity of the person"
    " MUST be preserved exactly."
)

# Used only by render_with_source_and_talent — states input ordering when BOTH
# source product AND talent reference images are attached simultaneously.
_INPUT_ORDER_NOTE = (
    "INPUT ORDER NOTE: Two reference images are attached. The FIRST input is"
    " the source product image (authority for product appearance). The SECOND"
    " input is the talent reference image (authority for talent identity)."
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

    The *preamble* argument (e.g. ``SOURCE_REFERENCE_PREAMBLE + "\\n\\n"``) is
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
    """Render the first-frame prompt for clip_index, prepending the talent reference preamble.

    Kept for backward compatibility. For new code, prefer render_with_source()
    or render_with_source_and_talent().

    For clip i ≥ 1, the caller should attach the clip-0 PNG bytes as the first
    image input alongside this prompt. The preamble instructs the model to anchor
    person identity and product appearance on the reference image; only scene,
    pose, framing, and lighting may vary.
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
    """Return a softened, talent-reference-anchored variant of the first-frame prompt.

    Kept for backward compatibility. Combines render_with_reference() with
    heuristic safety softening. Output order: safety preamble, reference
    instruction, softened body.
    """
    base_prompt = render(spec, brief, clip_index, talent_descriptor, product_size_hint)
    return _soften(base_prompt, preamble=REFERENCE_PREAMBLE + "\n\n")


# ---------------------------------------------------------------------------
# New dual-reference render functions (v1.3.0)
# ---------------------------------------------------------------------------


def render_with_source(
    spec: VideoSpec,
    brief: ProductBrief,
    clip_index: int,
    talent_descriptor: str,
    *,
    product_size_hint: str = _DEFAULT_SIZE_HINT,
) -> str:
    """Render with source product reference only (clip 0).

    Prepends SOURCE_REFERENCE_PREAMBLE so the model anchors on the actual
    product's appearance (labels, colour, shape) from the attached image.
    No talent reference is used — this is the clip-0 variant.

    Parameters
    ----------
    spec:
        The VideoSpec for the current video.
    brief:
        The ProductBrief describing the product.
    clip_index:
        Which clip (0-based) this first frame is for.
    talent_descriptor:
        Human-readable talent description resolved from the talent pool.
    product_size_hint:
        Optional override for the product scale description.

    Returns
    -------
    str
        The fully rendered prompt with SOURCE_REFERENCE_PREAMBLE prepended.
    """
    base = render(spec, brief, clip_index, talent_descriptor, product_size_hint=product_size_hint)
    return SOURCE_REFERENCE_PREAMBLE + "\n\n" + base


def render_with_source_and_talent(
    spec: VideoSpec,
    brief: ProductBrief,
    clip_index: int,
    talent_descriptor: str,
    *,
    product_size_hint: str = _DEFAULT_SIZE_HINT,
) -> str:
    """Render with both source product AND talent reference (clip ≥ 1).

    Prepends SOURCE_REFERENCE_PREAMBLE (for the first attached image) then
    REFERENCE_PREAMBLE (for the second attached image — clip 0's PNG), then
    the rendered base prompt body.

    Parameters
    ----------
    spec:
        The VideoSpec for the current video.
    brief:
        The ProductBrief describing the product.
    clip_index:
        Which clip (0-based) this first frame is for.
    talent_descriptor:
        Human-readable talent description resolved from the talent pool.
    product_size_hint:
        Optional override for the product scale description.

    Returns
    -------
    str
        The fully rendered prompt with both preambles prepended.
    """
    base = render(spec, brief, clip_index, talent_descriptor, product_size_hint=product_size_hint)
    return (
        _INPUT_ORDER_NOTE
        + "\n\n"
        + SOURCE_REFERENCE_PREAMBLE
        + "\n\n"
        + REFERENCE_PREAMBLE
        + "\n\n"
        + base
    )


def render_with_source_softened(
    spec: VideoSpec,
    brief: ProductBrief,
    clip_index: int,
    talent_descriptor: str,
    *,
    product_size_hint: str = _DEFAULT_SIZE_HINT,
) -> str:
    """Softened variant of render_with_source (clip 0 safety retry).

    SOURCE_REFERENCE_PREAMBLE is preserved verbatim; only the body is softened.
    Output order: safety preamble → source preamble → softened body.

    Parameters
    ----------
    spec:
        The VideoSpec for the current video.
    brief:
        The ProductBrief describing the product.
    clip_index:
        Which clip (0-based) this first frame is for.
    talent_descriptor:
        Human-readable talent description resolved from the talent pool.
    product_size_hint:
        Optional override for the product scale description.

    Returns
    -------
    str
        The fully rendered, softened, source-reference-anchored prompt.
    """
    base = render(spec, brief, clip_index, talent_descriptor, product_size_hint=product_size_hint)
    return _soften(base, preamble=SOURCE_REFERENCE_PREAMBLE + "\n\n")


def render_with_source_and_talent_softened(
    spec: VideoSpec,
    brief: ProductBrief,
    clip_index: int,
    talent_descriptor: str,
    *,
    product_size_hint: str = _DEFAULT_SIZE_HINT,
) -> str:
    """Softened variant of render_with_source_and_talent (clip ≥1 safety retry).

    Both preambles are preserved verbatim; only the body is softened.
    Output order: safety preamble → source preamble → talent preamble → softened body.

    Parameters
    ----------
    spec:
        The VideoSpec for the current video.
    brief:
        The ProductBrief describing the product.
    clip_index:
        Which clip (0-based) this first frame is for.
    talent_descriptor:
        Human-readable talent description resolved from the talent pool.
    product_size_hint:
        Optional override for the product scale description.

    Returns
    -------
    str
        The fully rendered, softened, dual-reference-anchored prompt.
    """
    base = render(spec, brief, clip_index, talent_descriptor, product_size_hint=product_size_hint)
    return _soften(
        base,
        preamble=(
            _INPUT_ORDER_NOTE
            + "\n\n"
            + SOURCE_REFERENCE_PREAMBLE
            + "\n\n"
            + REFERENCE_PREAMBLE
            + "\n\n"
        ),
    )
