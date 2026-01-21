"""Typography variant taxonomy and classification."""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class VariantMetadata:
    """Parsed metadata for a typography variant."""
    variant_id: str
    variant_bucket: str  # "font_family" | "emphasis" | "capitalization"
    font_family: str  # "Times" | "Arial" | "Comic" | "Monospace" | "OpenDyslexic"
    weight: str  # "regular" | "bold"
    caps: str  # "normal" | "all_caps"
    size: int  # Font size in points


# Variant ID to metadata mapping
VARIANT_TAXONOMY: Dict[str, VariantMetadata] = {
    # Font family variants (baseline: regular weight, normal case, standard size 48)
    "T1_times_regular": VariantMetadata(
        variant_id="T1_times_regular",
        variant_bucket="font_family",
        font_family="Times",
        weight="regular",
        caps="normal",
        size=48,
    ),
    "T3_arial_regular": VariantMetadata(
        variant_id="T3_arial_regular",
        variant_bucket="font_family",
        font_family="Arial",
        weight="regular",
        caps="normal",
        size=48,
    ),
    "T6_monospace": VariantMetadata(
        variant_id="T6_monospace",
        variant_bucket="font_family",
        font_family="Monospace",
        weight="regular",
        caps="normal",
        size=48,
    ),
    "T7_comic": VariantMetadata(
        variant_id="T7_comic",
        variant_bucket="font_family",
        font_family="Comic",
        weight="regular",
        caps="normal",
        size=48,
    ),
    "A1_opendyslexic_regular": VariantMetadata(
        variant_id="A1_opendyslexic_regular",
        variant_bucket="font_family",
        font_family="OpenDyslexic",
        weight="regular",
        caps="normal",
        size=48,
    ),

    # Emphasis variants (bold vs regular, same font family)
    "T2_times_bold": VariantMetadata(
        variant_id="T2_times_bold",
        variant_bucket="emphasis",
        font_family="Times",
        weight="bold",
        caps="normal",
        size=48,
    ),
    "T4_arial_bold": VariantMetadata(
        variant_id="T4_arial_bold",
        variant_bucket="emphasis",
        font_family="Arial",
        weight="bold",
        caps="normal",
        size=48,
    ),

    # Capitalization variant (Arial lowercase vs ALL CAPS)
    "T5_arial_all_caps": VariantMetadata(
        variant_id="T5_arial_all_caps",
        variant_bucket="capitalization",
        font_family="Arial",
        weight="bold",
        caps="all_caps",
        size=48,
    ),

    # T8_small_text is tricky - it's size manipulation, could be "size" bucket or "emphasis"
    # Placing in emphasis for now since it affects readability/urgency similar to bold
    "T8_small_text": VariantMetadata(
        variant_id="T8_small_text",
        variant_bucket="emphasis",
        font_family="Arial",
        weight="regular",
        caps="normal",
        size=28,
    ),
}


def get_variant_metadata(variant_id: str) -> Optional[VariantMetadata]:
    """Get metadata for a variant ID."""
    return VARIANT_TAXONOMY.get(variant_id)


def classify_variant_bucket(variant_id: str) -> str:
    """
    Classify a variant into one of three buckets:
    - font_family: Different typeface (Times, Arial, Comic, etc.)
    - emphasis: Bold vs regular, or size changes
    - capitalization: ALL CAPS vs normal case
    """
    metadata = get_variant_metadata(variant_id)
    if metadata is None:
        return "unknown"
    return metadata.variant_bucket


# Mapping for long-format CSV export
def get_variant_attributes(variant_id: str) -> Dict[str, Optional[str]]:
    """
    Extract all variant attributes for CSV export.

    Returns dict with keys:
    - variant_bucket
    - font_family
    - weight
    - caps
    """
    if variant_id == "__text__":
        return {
            "variant_bucket": None,
            "font_family": None,
            "weight": None,
            "caps": None,
        }

    metadata = get_variant_metadata(variant_id)
    if metadata is None:
        return {
            "variant_bucket": "unknown",
            "font_family": None,
            "weight": None,
            "caps": None,
        }

    return {
        "variant_bucket": metadata.variant_bucket,
        "font_family": metadata.font_family,
        "weight": metadata.weight,
        "caps": metadata.caps,
    }
