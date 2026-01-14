"""OCR utilities using Tesseract."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Try to import pytesseract, but allow graceful degradation
try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed. OCR functionality will be disabled.")


def check_tesseract_available() -> bool:
    """Check if Tesseract is available and working."""
    if not TESSERACT_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_text(image_path: Path, lang: str = "eng") -> str:
    """Extract text from image using Tesseract OCR."""
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract is not installed. Run: pip install pytesseract")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang).strip()
    return text


def run_ocr_on_sentences(
    sentences_df: pd.DataFrame,
    rendered_dir: Path,
    ocr_output_dir: Path,
    lang: str = "eng",
    reference_variant: str = "T3_arial_regular",
) -> pd.DataFrame:
    """
    Run OCR on rendered sentence images.

    Uses a single reference variant to generate deduped OCR baselines.
    Writes per-sentence OCR text files.

    Returns DataFrame with sentence_id and ocr_text.
    """
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract is not installed")

    ocr_output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for _, row in tqdm(sentences_df.iterrows(), total=len(sentences_df), desc="Running OCR"):
        sentence_id = row["sentence_id"]
        sentence_dir = rendered_dir / f"sentence_{sentence_id:03d}"

        # Use reference variant for OCR baseline (deduped across variants)
        image_path = sentence_dir / f"{reference_variant}.png"

        # Fall back to first available variant if reference doesn't exist
        if not image_path.exists():
            available = list(sentence_dir.glob("*.png"))
            if available:
                image_path = available[0]
                logger.warning(
                    f"Reference variant {reference_variant} not found for sentence {sentence_id}, "
                    f"using {image_path.name}"
                )
            else:
                logger.error(f"No rendered images found for sentence {sentence_id}")
                continue

        # Check if OCR already done
        ocr_file = ocr_output_dir / f"sentence_{sentence_id:03d}.txt"

        if ocr_file.exists():
            ocr_text = ocr_file.read_text()
        else:
            ocr_text = extract_text(image_path, lang)
            ocr_file.write_text(ocr_text)

        results.append({
            "sentence_id": sentence_id,
            "ocr_text": ocr_text,
            "ocr_path": str(ocr_file),
        })

    return pd.DataFrame(results)


def run_ocr_on_artifacts(
    artifacts_df: pd.DataFrame,
    rendered_dir: Path,
    ocr_output_dir: Path,
    lang: str = "eng",
    reference_variant: str = "T3_arial_regular",
) -> pd.DataFrame:
    """
    Run OCR on rendered artifact images.

    Uses a single reference variant to generate deduped OCR baselines.
    Writes per-artifact OCR text files.

    Returns DataFrame with artifact_id and ocr_text.
    """
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract is not installed")

    ocr_output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for _, row in tqdm(artifacts_df.iterrows(), total=len(artifacts_df), desc="Running OCR"):
        artifact_id = row["artifact_id"]
        artifact_dir = rendered_dir / artifact_id

        # Use reference variant for OCR baseline
        image_path = artifact_dir / f"{reference_variant}.png"

        # Fall back to first available variant
        if not image_path.exists():
            available = list(artifact_dir.glob("*.png"))
            if available:
                image_path = available[0]
            else:
                logger.error(f"No rendered images found for artifact {artifact_id}")
                continue

        # Check if OCR already done
        ocr_file = ocr_output_dir / f"{artifact_id}.txt"

        if ocr_file.exists():
            ocr_text = ocr_file.read_text()
        else:
            ocr_text = extract_text(image_path, lang)
            ocr_file.write_text(ocr_text)

        results.append({
            "artifact_id": artifact_id,
            "ocr_text": ocr_text,
            "ocr_path": str(ocr_file),
        })

    return pd.DataFrame(results)


def run_ocr_per_variant(
    sentence_id: int,
    rendered_dir: Path,
    ocr_output_dir: Path,
    lang: str = "eng",
) -> pd.DataFrame:
    """
    Run OCR on all variants for a single sentence (for QC purposes).

    Returns DataFrame with variant_id and ocr_text.
    """
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract is not installed")

    sentence_dir = rendered_dir / f"sentence_{sentence_id:03d}"
    variant_ocr_dir = ocr_output_dir / f"sentence_{sentence_id:03d}"
    variant_ocr_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for image_path in sentence_dir.glob("*.png"):
        variant_id = image_path.stem
        ocr_file = variant_ocr_dir / f"{variant_id}.txt"

        if ocr_file.exists():
            ocr_text = ocr_file.read_text()
        else:
            ocr_text = extract_text(image_path, lang)
            ocr_file.write_text(ocr_text)

        results.append({
            "sentence_id": sentence_id,
            "variant_id": variant_id,
            "ocr_text": ocr_text,
        })

    return pd.DataFrame(results)
