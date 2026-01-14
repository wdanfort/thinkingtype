"""Reporting utilities for generating manifests and summaries."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from typo_eval.config import TypoEvalConfig, config_to_dict


def get_git_commit() -> str:
    """Get current git commit hash."""
    git_head = Path(".git/HEAD")
    if not git_head.exists():
        return ""
    ref = git_head.read_text().strip()
    if ref.startswith("ref:"):
        ref_path = Path(".git") / ref.replace("ref: ", "")
        if ref_path.exists():
            return ref_path.read_text().strip()
    return ref


def get_system_info() -> Dict[str, str]:
    """Get system information."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "machine": platform.machine(),
    }


def get_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_manifest(
    run_id: str,
    config: TypoEvalConfig,
    repo_root: Optional[Path] = None,
    total_planned_calls: int = 0,
) -> Dict[str, Any]:
    """
    Generate run manifest with all metadata for reproducibility.

    Includes:
    - Config copy
    - Git commit hash
    - Python version, OS info
    - Fonts list with file hashes
    - Total planned calls
    """
    from typo_eval.config import get_repo_root, resolve_font_path

    root = repo_root or get_repo_root()

    # Build fonts manifest with hashes
    fonts_manifest = {}
    for variant in config.typography_variants:
        font_path = resolve_font_path(variant.font_file, root)
        fonts_manifest[variant.font_file] = {
            "variant_id": variant.id,
            "path": str(font_path),
            "hash": get_file_hash(font_path) if font_path.exists() else "",
            "exists": font_path.exists(),
        }

    manifest = {
        "run_id": run_id,
        "created_at": dt.datetime.utcnow().isoformat(),
        "config": config_to_dict(config),
        "git_commit": get_git_commit(),
        "system_info": get_system_info(),
        "fonts": fonts_manifest,
        "total_planned_calls": total_planned_calls,
    }

    return manifest


def write_manifest(manifest: Dict[str, Any], output_path: Path) -> None:
    """Write manifest to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load manifest from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text())
