"""Configuration loading utilities for the UGC pipeline.

Loads `pipeline_config.yaml` and `talent_pool.yaml` from the `config/` directory,
falling back to the `*.example.yaml` variants when the real files are absent.
See SPEC.md §9 for the expected shape of each file.
"""

from __future__ import annotations

import pathlib
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Default config directory (relative to project root, not this file's location)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_DIR = pathlib.Path("config")


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------


def load_pipeline_config(path: pathlib.Path | None = None) -> dict[str, Any]:
    """Load the pipeline configuration YAML and return it as a plain dict.

    Resolution order:
    1. *path* if explicitly provided.
    2. `config/pipeline_config.yaml` (the real, gitignored operational file).
    3. `config/pipeline_config.example.yaml` (tracked fallback for contributors).

    Raises `FileNotFoundError` if neither file is found.
    """
    if path is not None:
        candidates = [path]
    else:
        candidates = [
            _DEFAULT_CONFIG_DIR / "pipeline_config.yaml",
            _DEFAULT_CONFIG_DIR / "pipeline_config.example.yaml",
        ]

    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}

    raise FileNotFoundError(
        "Pipeline configuration file not found. "
        "Expected one of: "
        + ", ".join(str(c) for c in candidates)
        + ". Copy config/pipeline_config.example.yaml to config/pipeline_config.yaml "
        "and adjust values."
    )


# ---------------------------------------------------------------------------
# Talent pool
# ---------------------------------------------------------------------------


def load_talent_pool(path: pathlib.Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the talent pool YAML and return a mapping of talent_id -> descriptor dict.

    Resolution order:
    1. *path* if explicitly provided.
    2. `config/talent_pool.yaml` (gitignored, project-specific).
    3. `config/talent_pool.example.yaml` (tracked generic example).

    Raises `FileNotFoundError` if neither file is found.
    """
    if path is not None:
        candidates = [path]
    else:
        candidates = [
            _DEFAULT_CONFIG_DIR / "talent_pool.yaml",
            _DEFAULT_CONFIG_DIR / "talent_pool.example.yaml",
        ]

    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            return {str(k): dict(v) for k, v in data.items()}

    raise FileNotFoundError(
        "Talent pool configuration file not found. "
        "Expected one of: "
        + ", ".join(str(c) for c in candidates)
        + ". Copy config/talent_pool.example.yaml to config/talent_pool.yaml "
        "and add project-specific talent descriptors."
    )


# ---------------------------------------------------------------------------
# Brand guidance (optional)
# ---------------------------------------------------------------------------


def load_brand_guidance(path: pathlib.Path | None = None) -> dict[str, Any]:
    """Load optional brand guidance YAML and return it as a plain dict.

    Resolution order:
    1. *path* if explicitly provided (FileNotFoundError if missing).
    2. ``config/brand_guidance.yaml`` (gitignored, real positioning).
    3. Empty dict if neither file exists — brand guidance is opt-in.

    The returned dict is passed to the creative director and first-frame
    prompts so they can incorporate the project's positioning, hard rules,
    and behavioural directions without leaking domain-specific language
    into tracked source files.
    """
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"Brand guidance file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    candidate = _DEFAULT_CONFIG_DIR / "brand_guidance.yaml"
    if candidate.is_file():
        with candidate.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


# ---------------------------------------------------------------------------
# Talent descriptor rendering
# ---------------------------------------------------------------------------


def get_talent_descriptor(talent_id: str, pool: dict[str, dict[str, Any]]) -> str:
    """Render a one-line English description for *talent_id* from the talent *pool*.

    The rendered string is used in Creative Director and First-Frame Composite prompts.
    Format: "<gender>, <age_range>, <aesthetic>, <camera_relationship>, <lighting_preference>"

    Raises `KeyError` if *talent_id* is not found in *pool*.
    """
    if talent_id not in pool:
        raise KeyError(
            f"talent_id {talent_id!r} not found in talent pool. "
            f"Available: {sorted(pool.keys())}"
        )
    desc = pool[talent_id]
    parts = [
        desc.get("gender", ""),
        desc.get("age_range", ""),
        desc.get("aesthetic", ""),
        desc.get("camera_relationship", ""),
        desc.get("lighting_preference", ""),
    ]
    return ", ".join(p for p in parts if p)
