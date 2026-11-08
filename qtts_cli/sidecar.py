"""Reproducibility sidecar: ``<audio>.recipe.json`` (FR-P0-06, PRD §8.4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .utils import atomic_write_json, sha256_text

SCHEMA_VERSION = 1


def recipe_path(audio_path: str | Path) -> Path:
    """``out.wav`` -> ``out.recipe.json`` (PRD acceptance; codex P0-6)."""
    return Path(audio_path).with_suffix(".recipe.json")


def build_recipe(
    *,
    qtts_version: str,
    text: str,
    source: str,
    voice_ref: str,
    voice: Optional[dict[str, Any]],
    model: dict[str, Any],
    sampling: dict[str, Any],
    segments: list[dict[str, Any]],
    created_at: str,
    seed: Optional[int] = None,
    normalization: Optional[dict[str, Any]] = None,
    device: Optional[str] = None,
    instruct: Optional[str] = None,
    language: Optional[str] = None,
    subtitle: Optional[dict[str, Any]] = None,
    metrics: Optional[dict[str, Any]] = None,
    postprocess: Optional[dict[str, Any]] = None,
    timing: Optional[dict[str, Any]] = None,
    backend: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    recipe: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "qtts_version": qtts_version,
        "created_at": created_at,
        # PRD §9 compliance: AI-generated content is marked in the recipe.
        "ai_generated": True,
        "generator": "qtts",
        "input": {
            "text_sha256": sha256_text(text),
            "chars": len(text),
            "source": source,
        },
        "voice": {
            "ref": voice_ref,
            "prompt_sha256": (voice or {}).get("prompt_sha256"),
            "ref_audio_sha256": (voice or {}).get("ref_audio_sha256"),
        },
        "model": {
            "id": model.get("id"),
            "revision": model.get("revision"),
            "fingerprint": model.get("fingerprint"),
            "dtype": model.get("dtype"),
            "path": model.get("path"),
        },
        "language": language,
        "instruct": instruct,
        "seed": seed,
        "normalization": normalization or {},
        "sampling": sampling,
        "segments": segments,
        "postprocess": postprocess or {},
        "metrics": metrics or {},
        "timing": timing or {},
    }
    if backend is not None:
        recipe["backend"] = backend
    if device:
        recipe["device"] = device
    if subtitle is not None:
        recipe["subtitle"] = subtitle
    return recipe


def write_recipe(audio_path: str | Path, recipe: dict[str, Any]) -> Path:
    path = recipe_path(audio_path)
    atomic_write_json(path, recipe)
    return path
