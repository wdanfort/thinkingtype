"""Rendering utilities for text images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from typo_eval.config import RenderConfig, TypoEvalConfig, TypographyVariant, resolve_font_path

logger = logging.getLogger(__name__)


def wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> List[str]:
    """Wrap text to fit within max_width."""
    words = text.split()
    lines: List[str] = []
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


def render_text_image(
    text: str,
    font_path: Path,
    font_size: int,
    uppercase: bool,
    render_cfg: RenderConfig,
    line_spacing: float = 1.25,
) -> Image.Image:
    """
    Render text as an image with container framing.

    Uses container styling to prevent clipping issues (the "poster effect" fix).
    """
    display_text = text.upper() if uppercase else text

    # Container styling from config
    container = render_cfg.container
    pad_x = container.get("padding", 80)
    pad_y = render_cfg.margins
    radius = container.get("radius", 24)
    fill = tuple(container.get("fill", [245, 246, 248]))
    outline = tuple(container.get("outline", [220, 223, 228]))
    outline_width = 3

    # Create temporary canvas for measurement
    temp_img = Image.new("RGB", (render_cfg.image_width, 1200), "white")
    draw = ImageDraw.Draw(temp_img)

    font = ImageFont.truetype(str(font_path), font_size)
    lines = wrap_text(display_text, font, render_cfg.max_text_width, draw)

    # Calculate line height
    line_bbox = draw.textbbox((0, 0), "Ag", font=font)
    base_line_height = line_bbox[3] - line_bbox[1]
    line_height = int(base_line_height * line_spacing)

    # Calculate max line width
    max_line_w = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        max_line_w = max(max_line_w, bbox[2] - bbox[0])

    text_block_h = line_height * len(lines)

    # Container size
    container_w = min(
        render_cfg.image_width - 2 * pad_x,
        max_line_w + 2 * pad_x,
    )
    container_h = text_block_h + 2 * pad_y

    # Final image
    total_height = container_h + 80
    img = Image.new("RGB", (render_cfg.image_width, total_height), "white")
    draw = ImageDraw.Draw(img)

    # Container position (centered)
    cx0 = (render_cfg.image_width - container_w) // 2
    cy0 = (total_height - container_h) // 2
    cx1 = cx0 + container_w
    cy1 = cy0 + container_h

    # Draw rounded rectangle container
    draw.rounded_rectangle(
        (cx0, cy0, cx1, cy1),
        radius=radius,
        fill=fill,
        outline=outline,
        width=outline_width,
    )

    # Draw text centered in container
    y = cy0 + (container_h - text_block_h) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        x = (render_cfg.image_width - w) // 2
        draw.text((x, y), line, fill="black", font=font)
        y += line_height

    return img


def render_sentences(
    config: TypoEvalConfig,
    sentences_df: pd.DataFrame,
    output_dir: Path,
    repo_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Render all sentences for all typography variants.

    Returns metadata DataFrame with paths to rendered images.
    """
    metadata_rows = []

    for _, row in tqdm(sentences_df.iterrows(), total=len(sentences_df), desc="Rendering sentences"):
        sentence_id = row["sentence_id"]
        text = row["text"]
        category = row.get("category", "")

        sentence_dir = output_dir / f"sentence_{sentence_id:03d}"
        sentence_dir.mkdir(parents=True, exist_ok=True)

        for variant in config.typography_variants:
            font_path = resolve_font_path(variant.font_file, repo_root)

            if not font_path.exists():
                logger.warning(f"Font not found: {font_path}, skipping variant {variant.id}")
                continue

            image_path = sentence_dir / f"{variant.id}.png"

            if not image_path.exists():
                img = render_text_image(
                    text,
                    font_path,
                    variant.size,
                    variant.uppercase,
                    config.render,
                    variant.line_spacing,
                )
                img.save(image_path)

            metadata_rows.append({
                "sentence_id": sentence_id,
                "category": category,
                "variant_id": variant.id,
                "image_path": str(image_path),
            })

    return pd.DataFrame(metadata_rows)


def render_artifacts(
    config: TypoEvalConfig,
    artifacts_df: pd.DataFrame,
    output_dir: Path,
    repo_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Render all artifacts for all typography variants.

    Returns metadata DataFrame with paths to rendered images.
    """
    metadata_rows = []

    for _, row in tqdm(artifacts_df.iterrows(), total=len(artifacts_df), desc="Rendering artifacts"):
        artifact_id = row["artifact_id"]
        text = row["text"]
        artifact_type = row.get("type", "")

        artifact_dir = output_dir / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for variant in config.typography_variants:
            font_path = resolve_font_path(variant.font_file, repo_root)

            if not font_path.exists():
                logger.warning(f"Font not found: {font_path}, skipping variant {variant.id}")
                continue

            image_path = artifact_dir / f"{variant.id}.png"

            if not image_path.exists():
                img = render_text_image(
                    text,
                    font_path,
                    variant.size,
                    variant.uppercase,
                    config.render,
                    variant.line_spacing,
                )
                img.save(image_path)

            metadata_rows.append({
                "artifact_id": artifact_id,
                "type": artifact_type,
                "variant_id": variant.id,
                "image_path": str(image_path),
            })

    return pd.DataFrame(metadata_rows)
