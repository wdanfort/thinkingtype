"""Fontset utilities for vendored fonts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_fontset(fontset_path: str | Path) -> dict[str, Any]:
    path = Path(fontset_path)
    if not path.exists():
        raise FileNotFoundError(f"Fontset manifest not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text())
    else:
        data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Fontset manifest must be a JSON/YAML object.")
    return data


def _resolve_font_file(font_path: Path) -> Path | None:
    if font_path.exists():
        return font_path
    parent = font_path.parent
    if not parent.exists():
        return None
    candidates = list(parent.glob("*.ttf")) + list(parent.glob("*.TTF"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def get_font_path(font_key: str, repo_root: Path, fontset: dict[str, Any]) -> Path:
    fonts = fontset.get("fonts", {})
    if font_key not in fonts:
        raise KeyError(f"Font key not found in fontset: {font_key}")
    font_entry = fonts[font_key]
    font_rel_path = font_entry.get("path")
    if not font_rel_path:
        raise ValueError(f"Font path missing for key: {font_key}")
    resolved = (repo_root / font_rel_path).resolve()
    fallback = _resolve_font_file(resolved)
    if fallback is None:
        return resolved
    return fallback


def validate_fontset(repo_root: Path, fontset: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    fonts = fontset.get("fonts", {})
    for font_key, entry in fonts.items():
        font_rel_path = entry.get("path")
        if not font_rel_path:
            missing.append(f"{font_key}: <missing path>")
            continue
        font_path = (repo_root / font_rel_path).resolve()
        if _resolve_font_file(font_path) is None:
            missing.append(str(font_path))
    return missing


def detect_fontset_mismatches(repo_root: Path, fontset: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    fonts = fontset.get("fonts", {})
    for font_key, entry in fonts.items():
        font_rel_path = entry.get("path")
        if not font_rel_path:
            continue
        font_path = (repo_root / font_rel_path).resolve()
        resolved = _resolve_font_file(font_path)
        if resolved is None:
            continue
        if resolved != font_path:
            mismatches.append(f"{font_key}: {font_path.name} -> {resolved.name}")
    return mismatches


def load_fontset_paths(repo_root: Path, fontset_path: str | Path) -> dict[str, Path]:
    """Notebook-friendly helper returning absolute paths keyed by font key."""
    fontset = load_fontset(fontset_path)
    paths: dict[str, Path] = {}
    for font_key in fontset.get("fonts", {}):
        paths[font_key] = get_font_path(font_key, repo_root, fontset)
    return paths
