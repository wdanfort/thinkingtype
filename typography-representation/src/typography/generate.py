"""Artifact generation logic."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from typography.config import TypographyConfig
from typography.io import ensure_dir
from typography.render import fallback_font_path, render_text_image, resolve_font_path


def generate_artifacts(config: TypographyConfig, logger) -> Path:
    sentences = pd.read_csv(config.sentences_csv)
    variants = pd.read_csv(config.variants_csv)

    artifact_dir = Path(config.output_dir)
    text_dir = artifact_dir / "text"
    image_dir = artifact_dir / "images"
    ensure_dir(text_dir)
    ensure_dir(image_dir)

    metadata_rows: List[dict] = []

    for _, row in sentences.iterrows():
        item_id = str(row["item_id"])
        text = row["text"]
        category = row["category"]
        text_path = text_dir / f"{item_id}.txt"
        if not text_path.exists():
            text_path.write_text(text)
        metadata_rows.append(
            {
                "artifact_set_id": config.artifact_set_id,
                "item_id": item_id,
                "category": category,
                "variant_id": "__text__",
                "representation": "text",
                "text_path": str(text_path),
                "image_path": "",
            }
        )

        item_image_dir = image_dir / item_id
        ensure_dir(item_image_dir)

        for _, variant in variants.iterrows():
            variant_id = str(variant["variant_id"])
            font_family = variant.get("font_family")
            font_path = variant.get("font_path")
            font_size = int(variant.get("font_size", 48))
            uppercase = bool(variant.get("uppercase", False))

            resolved_path, warning = resolve_font_path(font_family, font_path)
            if warning:
                logger.warning("%s Falling back to DejaVu Sans for variant %s", warning, variant_id)
                resolved_path = fallback_font_path()

            image_path = item_image_dir / f"{variant_id}.png"
            if not image_path.exists():
                img = render_text_image(text, resolved_path, font_size, uppercase, config.rendering)
                img.save(image_path)

            metadata_rows.append(
                {
                    "artifact_set_id": config.artifact_set_id,
                    "item_id": item_id,
                    "category": category,
                    "variant_id": variant_id,
                    "representation": "image",
                    "text_path": str(text_path),
                    "image_path": str(image_path),
                }
            )

    metadata_path = artifact_dir / "metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    logger.info("Artifacts written to %s", artifact_dir)
    return artifact_dir
