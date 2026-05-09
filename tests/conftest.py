"""Shared pytest configuration and fixtures for the UGC pipeline test suite.

Test layout:
  tests/unit/        — zero-cost unit tests; run on every commit
  tests/integration/ — paid Gemini text/vision tests; gated by UGC_RUN_PAID_TESTS=1
  tests/e2e/         — flagship end-to-end test with mocked Veo/Nano Banana

Integration tests must be decorated with `@pytest.mark.paid`. This conftest
registers the marker and provides an autouse skip fixture for the integration
directory so unpaid runs are not accidentally billed.
"""

from __future__ import annotations

import os
import pathlib
from typing import Generator

import pytest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "paid: mark test as requiring a live API key (set UGC_RUN_PAID_TESTS=1 to run)",
    )


# ---------------------------------------------------------------------------
# Paid-test guard
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def paid_tests_enabled() -> bool:
    """Return True if the UGC_RUN_PAID_TESTS environment variable is set to '1'."""
    return os.environ.get("UGC_RUN_PAID_TESTS") == "1"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip @pytest.mark.paid tests unless UGC_RUN_PAID_TESTS=1 is set."""
    if os.environ.get("UGC_RUN_PAID_TESTS") == "1":
        return
    skip_paid = pytest.mark.skip(reason="set UGC_RUN_PAID_TESTS=1 to run paid integration tests")
    for item in items:
        if item.get_closest_marker("paid"):
            item.add_marker(skip_paid)


# ---------------------------------------------------------------------------
# Sample product image fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_product_image_path() -> pathlib.Path:
    """Return path to a brand-neutral beige 1024x1024 PNG with 'SAMPLE BRAND' text.

    Generates the file on first call and caches it at `tests/fixtures/sample_product.png`.
    The `tests/fixtures/` directory is gitignored — this file is never tracked.
    """
    fixture_dir = pathlib.Path(__file__).parent / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    out_path = fixture_dir / "sample_product.png"

    if out_path.is_file():
        return out_path

    from PIL import Image, ImageDraw  # type: ignore[import-untyped]

    img = Image.new("RGB", (1024, 1024), color=(245, 240, 230))  # beige background
    draw = ImageDraw.Draw(img)
    # Draw text near the centre; default font is fine for a test fixture
    draw.text((400, 490), "SAMPLE BRAND", fill=(60, 50, 40))
    img.save(out_path, format="PNG")

    return out_path


# ---------------------------------------------------------------------------
# Clip fixture MP4 (generated via ffmpeg — used in E2E and veo tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def clip_fixture_mp4_path(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Return path to an 8-second 1080x1920 test MP4 generated via ffmpeg.

    Generated from an ffmpeg lavfi testsrc source. Cached at
    ``tests/fixtures/clip_fixture.mp4`` so repeated test runs avoid re-encoding.
    If the file already exists with size > 1 KB, it is reused.
    """
    fixture_dir = pathlib.Path(__file__).parent / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    out_path = fixture_dir / "clip_fixture.mp4"

    if out_path.is_file() and out_path.stat().st_size > 1024:
        return out_path

    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pytest.skip("imageio_ffmpeg not available — skipping clip fixture generation")

    import subprocess

    result = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=8:size=1080x1920:rate=30",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg clip fixture generation failed: {result.stderr.decode()[:200]}")

    return out_path


# ---------------------------------------------------------------------------
# First-frame fixture PNG (generated via Pillow)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def firstframe_fixture_png_path(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Return path to a 1080x1920 neutral gradient PNG for use as a first-frame fixture.

    Generated via Pillow. Cached at ``tests/fixtures/firstframe_fixture.png``.
    """
    fixture_dir = pathlib.Path(__file__).parent / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    out_path = fixture_dir / "firstframe_fixture.png"

    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path

    from PIL import Image  # type: ignore[import-untyped]

    # Create a neutral grey gradient (no text, brand-neutral)
    img = Image.new("RGB", (1080, 1920), color=(200, 200, 200))
    img.save(out_path, format="PNG")

    return out_path


# ---------------------------------------------------------------------------
# Pipeline workspace fixture (used by E2E test)
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_workspace(tmp_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """Return a dict of workspace paths for an E2E pipeline run.

    Returns
    -------
    dict with keys:
        ``state_root``: tmp_path / "state"
        ``artifacts_root``: tmp_path / "artifacts"
    """
    state_root = tmp_path / "state"
    artifacts_root = tmp_path / "artifacts"
    state_root.mkdir(parents=True)
    artifacts_root.mkdir(parents=True)
    return {"state_root": state_root, "artifacts_root": artifacts_root}
