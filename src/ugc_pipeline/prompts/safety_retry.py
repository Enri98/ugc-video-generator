"""Safety-retry rewrite prompt template for the UGC pipeline.

See SPEC.md §6.4 for the canonical template shape and an example rendering.
This prompt is sent to Gemini Flash (cheap text model) to rewrite a blocked
Veo structured prompt. No Italian content in the template itself — this is a
structural English prompt. The *input* passed to Flash may contain Italian
voiceover content inside the AUDIO section; Flash is instructed to preserve
the language while softening word choices.

Note: Unlike other steps, the Gemini Flash call for safety retry uses
``response_mime_type="text/plain"`` (not "application/json") because the output
is a plain rewritten string, not a structured JSON object.
"""

from __future__ import annotations

import hashlib

VERSION = "1.1.0"

TEMPLATE = """\
You are a prompt safety editor for a UGC video pipeline.
The following structured Veo prompt was rejected by the video generation model due to a safety filter.
It may contain SCENE: and AUDIO: sections, and the AUDIO section may contain quoted Italian voiceover content.
Rewrite ANY part of the prompt that may have triggered the safety filter — including the SCENE description
AND the Italian voiceover text inside the AUDIO section — to preserve the intended meaning while removing
any elements that might trigger content moderation.

Guidelines:
- Keep the product, talent, and setting.
- Remove or soften any physical contact, suggestive positioning, or ambiguous framing in the SCENE section.
- If the AUDIO section contains Italian voiceover text, soften any word choices that could be flagged,
  but keep the text in Italian and preserve the overall meaning.
- Preserve the structure: if SCENE: and AUDIO: section headers are present, keep them exactly as-is.
- Preserve the language of the script: Italian voiceover must remain Italian.
- Do not add new narrative elements not present in the original.
- Return the full structured prompt verbatim (not a fragment), no explanation.

ORIGINAL BLOCKED PROMPT:
"{original_prompt}"

REWRITTEN PROMPT:\
"""

CONTENT_SHA256 = hashlib.sha256(TEMPLATE.encode()).hexdigest()


def render(*, original_prompt: str) -> str:
    """Render the safety-retry rewrite prompt.

    Parameters
    ----------
    original_prompt:
        The full structured Veo prompt (SCENE + AUDIO sections) that was blocked
        by the video generation model's safety filter. May contain Italian voiceover
        content inside the AUDIO section.

    Returns
    -------
    str
        The fully rendered prompt to be sent to Gemini Flash for plain-text rewrite.
    """
    return TEMPLATE.format(original_prompt=original_prompt)
