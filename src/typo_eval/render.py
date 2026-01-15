"""Rendering utilities for text images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def normalize_font_sizes(
    font_paths: List[Path],
    requested_size: int,
) -> Dict[str, int]:
    """
    Normalize font sizes so all fonts render at the same visual height.

    Measures each font at the requested size, finds the maximum visual height,
    then scales each font's point size so they all render at that same height.

    Args:
        font_paths: List of font file paths to normalize
        requested_size: The requested point size (baseline for measurement)

    Returns:
        Dictionary mapping font path (as string) to normalized point size
    """
    if not font_paths:
        return {}

    # Measure visual heights at requested size
    visual_heights: Dict[str, int] = {}
    temp_img = Image.new("RGB", (1000, 1000), "white")
    draw = ImageDraw.Draw(temp_img)

    for font_path in font_paths:
        try:
            font = ImageFont.truetype(str(font_path), requested_size)
            # Measure visual height using "Ag" which includes ascenders/descenders
            bbox = draw.textbbox((0, 0), "Ag", font=font)
            visual_height = bbox[3] - bbox[1]
            visual_heights[str(font_path)] = visual_height
        except Exception as e:
            logger.warning(f"Could not measure font {font_path}: {e}")
            visual_heights[str(font_path)] = 0

    # Find max visual height
    max_height = max(visual_heights.values()) if visual_heights else requested_size

    # Calculate normalization factors
    normalized_sizes: Dict[str, int] = {}
    for font_path_str, current_height in visual_heights.items():
        if current_height > 0:
            # Scale: new_size = requested_size * (max_height / current_height)
            scale_factor = max_height / current_height
            normalized_size = max(1, int(round(requested_size * scale_factor)))
            normalized_sizes[font_path_str] = normalized_size
        else:
            normalized_sizes[font_path_str] = requested_size

    return normalized_sizes


def render_text_image(
    text: str,
    font_path: Path,
    font_size: int,
    uppercase: bool,
    render_cfg: RenderConfig,
    line_spacing: float = 1.25,
) -> Image.Image:
    """
    Render text as an image on a single line with container framing.

    All text is rendered on a single line for experiment consistency.
    """
    display_text = text.upper() if uppercase else text

    # Container styling from config
    container = render_cfg.container
    pad_x = container.get("padding", 80)
    pad_y = render_cfg.margins
    radius = container.get("radius", 24)
    fill = None  # Transparent - no fill color
    outline = None  # Transparent - no outline

    # Create temporary canvas for measurement
    temp_img = Image.new("RGB", (render_cfg.image_width, 1200), "white")
    draw = ImageDraw.Draw(temp_img)

    font = ImageFont.truetype(str(font_path), font_size)

    # Calculate line height from base font
    line_bbox = draw.textbbox((0, 0), "Ag", font=font)
    base_line_height = line_bbox[3] - line_bbox[1]
    line_height = int(base_line_height * line_spacing)

    # Calculate width of text on a single line
    text_bbox = draw.textbbox((0, 0), display_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]

    # Container size (single line) - dynamic width to fit all text
    container_w = text_width + 2 * pad_x
    container_h = line_height + 2 * pad_y

    # Final image dimensions - width expands to fit text
    image_width = container_w + 80  # Add margins on sides
    total_height = container_h + 80  # Add margins on top/bottom
    img = Image.new("RGB", (image_width, total_height), "white")
    draw = ImageDraw.Draw(img)

    # Container position (centered)
    cx0 = (image_width - container_w) // 2
    cy0 = (total_height - container_h) // 2
    cx1 = cx0 + container_w
    cy1 = cy0 + container_h

    # Draw rounded rectangle container (transparent - no fill or outline)
    draw.rounded_rectangle(
        (cx0, cy0, cx1, cy1),
        radius=radius,
        fill=fill,
        outline=outline,
    )

    # Draw text centered in container (single line)
    y = cy0 + (container_h - line_height) // 2
    bbox = draw.textbbox((0, 0), display_text, font=font)
    w = bbox[2] - bbox[0]
    x = (image_width - w) // 2
    draw.text((x, y), display_text, fill="black", font=font)

    return img


def render_sentences(
    config: TypoEvalConfig,
    sentences_df: pd.DataFrame,
    output_dir: Path,
    repo_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Render all sentences for all typography variants.

    Font sizes are normalized so all fonts at the same size level render
    at consistent visual heights for experimental consistency.

    Returns metadata DataFrame with paths to rendered images.
    """
    metadata_rows = []

    # Pre-compute normalized font sizes for all variants grouped by size
    size_to_normalized: Dict[int, Dict[str, int]] = {}
    for variant in config.typography_variants:
        if variant.size not in size_to_normalized:
            # Collect all font paths for this size
            font_paths = []
            for v in config.typography_variants:
                if v.size == variant.size:
                    font_path = resolve_font_path(v.font_file, repo_root)
                    if font_path.exists():
                        font_paths.append(font_path)

            # Normalize sizes for all fonts at this size level
            if font_paths:
                size_to_normalized[variant.size] = normalize_font_sizes(font_paths, variant.size)

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
                # Use normalized font size
                normalized_size = size_to_normalized.get(variant.size, {}).get(
                    str(font_path), variant.size
                )

                img = render_text_image(
                    text,
                    font_path,
                    normalized_size,
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

    Font sizes are normalized so all fonts at the same size level render
    at consistent visual heights for experimental consistency.

    Returns metadata DataFrame with paths to rendered images.
    """
    metadata_rows = []

    # Pre-compute normalized font sizes for all variants grouped by size
    size_to_normalized: Dict[int, Dict[str, int]] = {}
    for variant in config.typography_variants:
        if variant.size not in size_to_normalized:
            # Collect all font paths for this size
            font_paths = []
            for v in config.typography_variants:
                if v.size == variant.size:
                    font_path = resolve_font_path(v.font_file, repo_root)
                    if font_path.exists():
                        font_paths.append(font_path)

            # Normalize sizes for all fonts at this size level
            if font_paths:
                size_to_normalized[variant.size] = normalize_font_sizes(font_paths, variant.size)

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
                # Use normalized font size
                normalized_size = size_to_normalized.get(variant.size, {}).get(
                    str(font_path), variant.size
                )

                img = render_text_image(
                    text,
                    font_path,
                    normalized_size,
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
