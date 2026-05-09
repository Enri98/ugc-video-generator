"""Probe test for Vertex AI access via service account JSON.

Purpose: verify the credentials/service-account.json file has the right roles
to call Vertex AI for both text generation and image generation, BEFORE we
refactor the pipeline clients away from the API-key flow.

Does NOT use GOOGLE_API_KEY. Authenticates via the service account file and
explicit project + location (standard Vertex AI auth path).

Run with:
    $env:UGC_RUN_PAID_TESTS = "1"
    .venv\\Scripts\\python -m pytest tests/integration/test_vertex_service_account_probe.py -v -s
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest


SA_PATH = pathlib.Path("credentials/service-account.json")
LOCATION = "us-central1"

TEXT_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "imagen-3.0-fast-generate-001"


def _load_project_id() -> str:
    assert SA_PATH.is_file(), f"Service account JSON not found at {SA_PATH}"
    with SA_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    project_id = data.get("project_id")
    assert project_id, "service-account.json has no project_id"
    return project_id


def _build_client():
    """Build a Vertex genai client authenticated via the service account JSON."""
    from google import genai  # type: ignore[import-untyped]
    from google.oauth2 import service_account  # type: ignore[import-untyped]

    project_id = _load_project_id()
    credentials = service_account.Credentials.from_service_account_file(
        str(SA_PATH),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return genai.Client(
        vertexai=True,
        project=project_id,
        location=LOCATION,
        credentials=credentials,
    )


@pytest.mark.paid
def test_vertex_sa_text_generation() -> None:
    """Smoke test: SA can call a Gemini text model on Vertex in us-central1."""
    client = _build_client()
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents="Reply with exactly the word: OK",
    )
    text = (response.text or "").strip()
    print(f"\n[text probe] model={TEXT_MODEL} response={text!r}")
    assert text, "Vertex text call returned empty response"


@pytest.mark.paid
def test_vertex_sa_image_generation() -> None:
    """Smoke test: SA can call an Imagen model on Vertex in us-central1."""
    client = _build_client()
    response = client.models.generate_images(
        model=IMAGE_MODEL,
        prompt="A simple beige ceramic coffee mug on a wooden table, soft natural light.",
        config={"number_of_images": 1, "aspect_ratio": "1:1"},
    )
    images = getattr(response, "generated_images", None) or []
    print(f"\n[image probe] model={IMAGE_MODEL} images_returned={len(images)}")
    assert len(images) >= 1, "Vertex image call returned no images"
    img = images[0].image
    png_bytes = getattr(img, "image_bytes", None) or b""
    assert len(png_bytes) > 1024, f"Image bytes suspiciously small: {len(png_bytes)}"
