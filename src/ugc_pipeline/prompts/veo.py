"""Veo prompt template for the UGC pipeline.

See SPEC.md §6.4 (or equivalent) for the canonical template shape. Each clip
in a VideoSpec gets its own Veo prompt. Exactly ONE clip per video carries
spoken audio (the speaking_clip_index); the rest receive ambient-only audio.
Concentrating speech in a single clip avoids voice-drift across independent
Veo calls.

LANGUAGE CONVENTION: structural instructions are English; the spoken line
(script_block) is Italian, passed verbatim to the model inside quotes.
"""

from __future__ import annotations

import hashlib
import re

VERSION = "1.0.0"

# Main structural template — drives CONTENT_SHA256.
TEMPLATE = """\
SCENE: {scene_description}

AUDIO: {audio_directive}\
"""

CONTENT_SHA256 = hashlib.sha256(TEMPLATE.encode()).hexdigest()

# Audio directive sub-templates used inside render().
AUDIO_SPEAKING_TEMPLATE = (
    'Have the on-screen talent speak naturally in Italian, in a {tone} delivery style'
    ' — calm, authentic UGC pacing, like recommending something to a friend.'
    ' Spoken line (Italian, verbatim): "{script_block}".'
    " Keep delivery wholesome and brand-safe."
    " Include subtle ambient room sound under the voice."
)

AUDIO_SILENT_TEMPLATE = (
    "No spoken dialogue."
    " Subtle ambient sound matching the scene only"
    " — light room tone, soft natural sounds appropriate to the setting."
    " No music, no voice."
)

# Pre-compiled pattern to detect unreplaced {placeholder} tokens.
_PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def render(
    *,
    scene_description: str,
    tone: str,
    is_speaking_clip: bool,
    script_block: str = "",
) -> str:
    """Render a Veo prompt for a single clip.

    Parameters
    ----------
    scene_description:
        English cinematographic description for this clip
        (one of ``VideoSpec.scene_descriptions[i]``).
    tone:
        Narrative tone for the video (e.g. ``"warm storyteller"``).
        One of the three canonical tones defined in ``prompts.director.TONES``.
    is_speaking_clip:
        True only for the clip whose index equals ``spec.speaking_clip_index``.
        Exactly one clip per video should have this set to True.
    script_block:
        Italian voiceover line (``VideoSpec.script_blocks[speaking_clip_index]``).
        Required when ``is_speaking_clip=True``; ignored when False.

    Returns
    -------
    str
        The fully rendered Veo prompt string.

    Raises
    ------
    ValueError
        If ``is_speaking_clip`` is True and ``script_block`` is empty or
        contains only whitespace.
    """
    if is_speaking_clip:
        if not script_block or not script_block.strip():
            raise ValueError(
                "script_block must be a non-empty Italian voiceover line "
                "when is_speaking_clip=True."
            )
        audio_directive = AUDIO_SPEAKING_TEMPLATE.format(
            tone=tone,
            script_block=script_block.strip(),
        )
    else:
        audio_directive = AUDIO_SILENT_TEMPLATE

    return TEMPLATE.format(
        scene_description=scene_description,
        audio_directive=audio_directive,
    )
