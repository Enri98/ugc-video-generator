"""Product analyst prompt template for the UGC pipeline.

See SPEC.md §6.1 for the canonical template shape and the coffee-cup fixture
rendering. The image is passed separately as inline_data per SPEC.md §6
canonical invocation pattern; this template carries only the text block.
"""

from __future__ import annotations

import hashlib

VERSION = "1.0.0"

TEMPLATE = """\
You are a product analyst for a consumer brand's UGC video pipeline.
Examine the attached product image and extract structured attributes.
Do not invent features not visible in the image.
Do not mention any brand name, logo text, or trademark.
Describe the product as a generic lifestyle item.

Output fields:
- shape: geometric form of the object
- dominant_colours: up to 5 hex codes or descriptive colour names
- packaging_style: describe the outer packaging if visible
- inferred_category: broad lifestyle category, no brand names
- lifestyle_contexts: 3-5 plausible everyday use-case settings
- visual_notes: texture, transparency, reflectivity observations (optional)

Respond in valid JSON conforming to the ProductBrief schema.\
"""

CONTENT_SHA256 = hashlib.sha256(TEMPLATE.encode()).hexdigest()


def render() -> str:
    """Return the analyst prompt template verbatim.

    The image is injected separately as inline_data by the caller; no
    template variables are needed for this prompt body.
    """
    return TEMPLATE
