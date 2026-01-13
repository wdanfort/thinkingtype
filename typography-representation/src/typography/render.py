"""Rendering utilities for text images."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from typography.config import RenderingConfig
from typography.fontset import get_font_path


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}" if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def _clean_font_value(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    value_str = str(value).strip()
    return value_str or None


def _find_font_key_by_family(font_family: str, fontset: dict) -> Optional[str]:
    for font_key, entry in fontset.get("fonts", {}).items():
        family = entry.get("family")
        if isinstance(family, str) and family.lower() == font_family.lower():
            return font_key
    return None


def resolve_font_path(
    font_family: Optional[str],
    font_path: Optional[str],
    font_key: Optional[str],
    fontset: Optional[dict],
    repo_root: Optional[Path],
    allow_system_fonts: bool = False,
    default_font_key: str = "inter",
) -> Tuple[Optional[Path], Optional[str]]:
    font_path_clean = _clean_font_value(font_path)
    font_family_clean = _clean_font_value(font_family)
    font_key_clean = _clean_font_value(font_key)

    if font_path_clean:
        path = Path(font_path_clean)
        if path.exists():
            return path, None
        return None, f"Font path not found: {font_path_clean}"

    if font_key_clean and fontset and repo_root:
        try:
            path = get_font_path(font_key_clean, repo_root, fontset)
            if path.exists():
                return path, None
            return None, f"Vendored font missing for key {font_key_clean}: {path}"
        except (KeyError, ValueError) as exc:
            return None, str(exc)

    if font_family_clean and fontset and repo_root:
        font_key = _find_font_key_by_family(font_family_clean, fontset)
        if font_key:
            path = get_font_path(font_key, repo_root, fontset)
            if path.exists():
                return path, None
            return None, f"Vendored font missing for family {font_family_clean}: {path}"

    if font_family_clean and allow_system_fonts:
        path = font_manager.findfont(font_family_clean, fallback_to_default=False)
        if path:
            return Path(path), None
        return None, f"Font family not found in system registry: {font_family_clean}"

    if fontset and repo_root:
        try:
            path = get_font_path(default_font_key, repo_root, fontset)
            if path.exists():
                return path, f"Using default vendored font: {default_font_key}"
            return None, f"Default vendored font missing: {path}"
        except (KeyError, ValueError) as exc:
            return None, str(exc)

    return None, "No font_path provided and no vendored fontset loaded."


def fallback_font_path(fontset: dict, repo_root: Path, default_font_key: str = "inter") -> Path:
    return get_font_path(default_font_key, repo_root, fontset)


def render_text_image(
    text: str,
    font_path: str | Path,
    font_size: int,
    uppercase: bool,
    render_cfg: RenderingConfig,
) -> Image.Image:
    display_text = text.upper() if uppercase else text

    temp_img = Image.new("RGB", (render_cfg.canvas_width, 1200), render_cfg.background_color)
    draw = ImageDraw.Draw(temp_img)

    font = ImageFont.truetype(str(font_path), font_size)
    lines = wrap_text(display_text, font, render_cfg.max_text_width, draw)

    line_bbox = draw.textbbox((0, 0), "Ag", font=font)
    base_line_height = line_bbox[3] - line_bbox[1]
    line_height = int(base_line_height * render_cfg.line_spacing)

    max_line_w = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        max_line_w = max(max_line_w, bbox[2] - bbox[0])

    text_block_h = line_height * len(lines)

    container_w = min(
        render_cfg.canvas_width - 2 * render_cfg.container_pad_x,
        max_line_w + 2 * render_cfg.container_pad_x,
    )
    container_h = text_block_h + 2 * render_cfg.container_pad_y

    total_height = container_h + 80
    img = Image.new("RGB", (render_cfg.canvas_width, total_height), render_cfg.background_color)
    draw = ImageDraw.Draw(img)

    cx0 = (render_cfg.canvas_width - container_w) // 2
    cy0 = (total_height - container_h) // 2
    cx1 = cx0 + container_w
    cy1 = cy0 + container_h

    draw.rounded_rectangle(
        (cx0, cy0, cx1, cy1),
        radius=render_cfg.container_radius,
        fill=tuple(render_cfg.container_fill),
        outline=tuple(render_cfg.container_outline),
        width=render_cfg.container_outline_width,
    )

    y = cy0 + (container_h - text_block_h) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        x = (render_cfg.canvas_width - w) // 2
        draw.text((x, y), line, fill=render_cfg.text_color, font=font)
        y += line_height

    return img
