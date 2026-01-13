"""Font management and validation utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

from typo_eval.config import TypoEvalConfig, resolve_font_path


# Required font files that must be bundled in assets/fonts/
REQUIRED_FONTS = {
    "LiberationSerif-Regular.ttf": {
        "family": "Liberation Serif",
        "license": "OFL / GPL+exception",
        "category": "serif",
    },
    "LiberationSerif-Bold.ttf": {
        "family": "Liberation Serif",
        "license": "OFL / GPL+exception",
        "category": "serif",
    },
    "LiberationSans-Regular.ttf": {
        "family": "Liberation Sans",
        "license": "OFL / GPL+exception",
        "category": "sans",
    },
    "LiberationSans-Bold.ttf": {
        "family": "Liberation Sans",
        "license": "OFL / GPL+exception",
        "category": "sans",
    },
    "LiberationMono-Regular.ttf": {
        "family": "Liberation Mono",
        "license": "OFL / GPL+exception",
        "category": "mono",
    },
    "ComicNeue-Regular.ttf": {
        "family": "Comic Neue",
        "license": "OFL",
        "category": "comic",
    },
    "OpenDyslexic-Regular.otf": {
        "family": "OpenDyslexic",
        "license": "OFL",
        "category": "accessibility",
    },
}


def validate_fonts(config: TypoEvalConfig, repo_root: Optional[Path] = None) -> List[str]:
    """
    Validate that all fonts referenced in config exist.

    Returns list of missing font file paths.
    """
    missing: List[str] = []

    for variant in config.typography_variants:
        font_path = resolve_font_path(variant.font_file, repo_root)
        if not font_path.exists():
            missing.append(str(font_path))

    return missing


def validate_required_fonts(repo_root: Optional[Path] = None) -> List[str]:
    """
    Validate that all required bundled fonts exist.

    Returns list of missing font file paths.
    """
    from typo_eval.config import get_repo_root

    root = repo_root or get_repo_root()
    fonts_dir = root / "assets" / "fonts"
    missing: List[str] = []

    for font_file in REQUIRED_FONTS:
        font_path = fonts_dir / font_file
        if not font_path.exists():
            missing.append(str(font_path))

    return missing


def get_font_hash(font_path: Path) -> str:
    """Compute SHA256 hash of font file."""
    if not font_path.exists():
        return ""
    return hashlib.sha256(font_path.read_bytes()).hexdigest()


def get_fonts_manifest(config: TypoEvalConfig, repo_root: Optional[Path] = None) -> Dict[str, str]:
    """
    Get manifest of all fonts with their hashes.

    Returns dict mapping font filename to SHA256 hash.
    """
    manifest: Dict[str, str] = {}

    for variant in config.typography_variants:
        font_path = resolve_font_path(variant.font_file, repo_root)
        if font_path.exists():
            manifest[variant.font_file] = get_font_hash(font_path)

    return manifest


def check_fonts_or_exit(config: TypoEvalConfig, repo_root: Optional[Path] = None) -> None:
    """
    Validate fonts and exit with clear error if any are missing.
    """
    missing = validate_fonts(config, repo_root)

    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            f"Missing font files:\n{missing_list}\n\n"
            "Please ensure all font files are placed in assets/fonts/\n"
            "Required fonts (open-source) can be downloaded from:\n"
            "  - Liberation fonts: https://github.com/liberationfonts/liberation-fonts\n"
            "  - Comic Neue: https://comicneue.com/\n"
            "  - OpenDyslexic: https://opendyslexic.org/"
        )
