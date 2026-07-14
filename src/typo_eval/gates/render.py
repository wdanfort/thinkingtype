"""Multi-line document rendering for gate stimuli.

Renders each gate document (several labeled lines) as an image under a
visual variant spec covering font, size scale, color, highlighting, and
layout density. Font sizes are normalized across variant fonts (same method
as the v0 harness) before per-variant size scaling, so a variant's scale is
an intentional manipulation rather than a font-metrics artifact.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from typo_eval.config import resolve_font_path
from typo_eval.gates.config import GateRenderConfig, GatesConfig, GateVariantSpec
from typo_eval.render import measure_image, normalize_font_sizes

logger = logging.getLogger(__name__)


def _wrap_paragraph(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> List[str]:
    """Wrap one paragraph to max_width, preserving word order."""
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_gate_document(
    text: str,
    variant: GateVariantSpec,
    font_path: Path,
    font_size: int,
    render_cfg: GateRenderConfig,
) -> Image.Image:
    """Render a multi-line document under a visual variant."""
    display_text = text.upper() if variant.uppercase else text

    font = ImageFont.truetype(str(font_path), font_size)

    # Measure on a scratch canvas
    scratch = Image.new("RGB", (render_cfg.image_width, 200), "white")
    draw = ImageDraw.Draw(scratch)

    ag_bbox = draw.textbbox((0, 0), "Ag", font=font)
    base_line_height = ag_bbox[3] - ag_bbox[1]
    line_height = int(base_line_height * variant.line_spacing)

    max_text_width = min(
        render_cfg.max_text_width,
        render_cfg.image_width - 2 * render_cfg.padding_x,
    )

    lines: List[str] = []
    for paragraph in display_text.split("\n"):
        if paragraph.strip():
            lines.extend(_wrap_paragraph(paragraph, font, max_text_width, draw))
        else:
            lines.append("")

    total_height = len(lines) * line_height + 2 * render_cfg.padding_y
    img = Image.new(
        "RGB", (render_cfg.image_width, total_height), variant.bg_color
    )
    draw = ImageDraw.Draw(img)

    y = render_cfg.padding_y
    for line in lines:
        if line:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            if variant.align == "center":
                x = (render_cfg.image_width - line_width) // 2
            else:
                x = render_cfg.padding_x
            if variant.highlight_color:
                pad = max(2, base_line_height // 8)
                draw.rectangle(
                    (
                        x - pad,
                        y - pad,
                        x + line_width + pad,
                        y + base_line_height + pad,
                    ),
                    fill=variant.highlight_color,
                )
            draw.text((x, y), line, fill=variant.text_color, font=font)
        y += line_height

    return img


def render_gate_items(
    config: GatesConfig,
    items_df: pd.DataFrame,
    output_dir: Path,
    repo_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Render every gate item under every variant.

    Returns metadata with image paths and measured ink ratios.
    """
    # Normalize across the distinct fonts used by variants at base size
    # (or a pinned font set, for runs that must match cached renders)
    norm_files = config.render.normalize_fonts or [v.font_file for v in config.variants]
    font_paths = []
    for font_file in norm_files:
        fp = resolve_font_path(font_file, repo_root)
        if fp.exists():
            font_paths.append(fp)
        else:
            raise FileNotFoundError(f"Font not found: {fp}")
    normalized = normalize_font_sizes(
        sorted(set(font_paths)),
        config.render.base_size,
        method=config.render.font_normalization,
    )

    rows = []
    for _, row in tqdm(
        items_df.iterrows(), total=len(items_df), desc="Rendering gate documents"
    ):
        item_dir = output_dir / row["gate"] / row["item_id"]
        item_dir.mkdir(parents=True, exist_ok=True)

        for variant in config.variants:
            font_path = resolve_font_path(variant.font_file, repo_root)
            base = normalized.get(str(font_path), config.render.base_size)
            font_size = max(1, int(round(base * variant.size_scale)))

            image_path = item_dir / f"{variant.id}.png"
            if not image_path.exists():
                if getattr(variant, "chrome", None):
                    from typo_eval.gates.chrome_render import render_chrome_document

                    img = render_chrome_document(
                        row["text"], variant, font_path, font_size, config.render
                    )
                else:
                    img = render_gate_document(
                        row["text"], variant, font_path, font_size, config.render
                    )
                img.save(image_path)

            width, height, ink_ratio = measure_image(image_path)
            rows.append(
                {
                    "gate": row["gate"],
                    "item_id": row["item_id"],
                    "scenario": row["scenario"],
                    "level": row["level"],
                    "variant_id": variant.id,
                    "image_path": str(image_path),
                    "font_size": font_size,
                    "image_width": width,
                    "image_height": height,
                    "ink_ratio": ink_ratio,
                }
            )

    return pd.DataFrame(rows)
