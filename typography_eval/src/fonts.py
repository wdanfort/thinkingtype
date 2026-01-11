from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List

from typography_eval import config


def find_font(path_pattern: str) -> str:
    matches = glob.glob(path_pattern)
    if not matches:
        return ""
    return matches[0]


def typography_variants() -> Dict[str, dict]:
    variants = {
        "T1_times_regular": {
            "font_family": "Liberation Serif",
            "font_path": find_font("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.*"),
            "size": 48,
            "uppercase": False,
            "variant_group": "baseline",
        },
        "T2_times_bold": {
            "font_family": "Liberation Serif",
            "font_path": find_font("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.*"),
            "size": 48,
            "uppercase": False,
            "variant_group": "baseline",
        },
        "T3_arial_regular": {
            "font_family": "Liberation Sans",
            "font_path": find_font("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.*"),
            "size": 48,
            "uppercase": False,
            "variant_group": "baseline",
        },
        "T4_arial_bold": {
            "font_family": "Liberation Sans",
            "font_path": find_font("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.*"),
            "size": 48,
            "uppercase": False,
            "variant_group": "baseline",
        },
        "T5_arial_all_caps": {
            "font_family": "Liberation Sans",
            "font_path": find_font("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.*"),
            "size": 48,
            "uppercase": True,
            "variant_group": "extreme",
        },
        "T6_monospace": {
            "font_family": "Liberation Mono",
            "font_path": find_font("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.*"),
            "size": 48,
            "uppercase": False,
            "variant_group": "baseline",
        },
        "T7_comic": {
            "font_family": "Comic Neue",
            "font_path": find_font("/usr/share/fonts/opentype/comic-neue/ComicNeue-Regular.*"),
            "size": 48,
            "uppercase": False,
            "variant_group": "extreme",
        },
        "T8_small_text": {
            "font_family": "Liberation Sans",
            "font_path": find_font("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.*"),
            "size": 28,
            "uppercase": False,
            "variant_group": "extreme",
        },
    }

    opendyslexic_path = "/usr/share/fonts/opentype/opendyslexic/OpenDyslexicAlta-Regular.otf"
    variants["A1_opendyslexic_regular"] = {
        "font_family": "OpenDyslexic",
        "font_path": opendyslexic_path if Path(opendyslexic_path).exists() else "",
        "size": 48,
        "uppercase": False,
        "variant_group": "accessibility",
    }

    return variants


def variants_table() -> List[dict]:
    rows = []
    for variant_id, cfg in typography_variants().items():
        rows.append(
            {
                "variant_id": variant_id,
                "font_family": cfg["font_family"],
                "font_path": cfg.get("font_path", ""),
                "size": cfg["size"],
                "uppercase": cfg["uppercase"],
                "variant_group": cfg["variant_group"],
                "variant_set_id": config.VARIANT_SET_ID,
            }
        )
    return rows
