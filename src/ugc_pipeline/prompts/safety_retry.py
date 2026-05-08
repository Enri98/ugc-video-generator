"""Safety-retry rewrite prompt template for the UGC pipeline.

See SPEC.md §6.4 for the canonical template shape and an example rendering.
This prompt is sent to Gemini Flash (cheap text model) to rewrite a blocked
Veo scene prompt. No Italian content — this is a structural English prompt.

Note: Unlike other steps, the Gemini Flash call for safety retry uses
``response_mime_type="text/plain"`` (not "application/json") because the output
is a plain rewritten string, not a structured JSON object.
"""

from __future__ import annotations

import hashlib

VERSION = "1.0.0"

TEMPLATE = """\
You are a prompt safety editor for a UGC video pipeline.
The following scene prompt was rejected by the video generation model due to a safety filter.
Rewrite it to preserve the intended visual storytelling while removing any elements
that might trigger content moderation.

Guidelines:
- Keep the product, talent, and setting.
- Remove or soften any physical contact, suggestive positioning, or ambiguous framing.
- Do not add new narrative elements not present in the original.
- Return only the rewritten prompt text, no explanation.

ORIGINAL BLOCKED PROMPT:
"{original_prompt}"

REWRITTEN PROMPT:\
"""

CONTENT_SHA256 = hashlib.sha256(TEMPLATE.encode()).hexdigest()


def render(original_prompt: str) -> str:
    """Render the safety-retry rewrite prompt.

    Parameters
    ----------
    original_prompt:
        The scene prompt that was blocked by the video generation model's safety filter.

    Returns
    -------
    str
        The fully rendered prompt to be sent to Gemini Flash for plain-text rewrite.
    """
    return TEMPLATE.format(original_prompt=original_prompt)
