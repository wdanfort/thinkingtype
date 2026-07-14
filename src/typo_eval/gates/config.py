"""Configuration schema for the gates pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field

from typo_eval.config import ProvidersConfig


class GateVariantSpec(BaseModel):
    """A visual presentation variant for gate documents.

    Extends the v0 typography variants with color, highlighting, and
    layout-density controls so typography is one cue among several.
    """

    id: str
    font_file: str = "LiberationSans-Regular.ttf"
    size_scale: float = 1.0  # multiplier on render.base_size (after normalization)
    line_spacing: float = 1.35
    uppercase: bool = False
    text_color: str = "#000000"
    bg_color: str = "#FFFFFF"
    highlight_color: Optional[str] = None  # filled band behind each text line
    align: str = "left"  # "left" | "center"
    # When set, render the document inside realistic app/forum UI chrome
    # instead of as a plain block: "forum_light" | "forum_dark" | "phone_light".
    chrome: Optional[str] = None


class GateRenderConfig(BaseModel):
    """Rendering configuration for multi-line gate documents."""

    image_width: int = 1400
    base_size: int = 40
    padding_x: int = 90
    padding_y: int = 70
    max_text_width: int = 1220
    font_normalization: str = "ag_height"
    # Fonts to include in size normalization. Defaults to the fonts used by
    # this config's variants, but follow-up experiments that reuse cached
    # gates_v1 images must pin this to the gates_v1 font set — otherwise a
    # smaller variant set normalizes to different point sizes and new renders
    # silently mismatch the cached ones.
    normalize_fonts: Optional[List[str]] = None


class CalibrationConfig(BaseModel):
    """Text-baseline calibration configuration."""

    repeats: int = 8
    # Acceptable text p(yes) band for an item to count as boundary-adjacent.
    band: Tuple[float, float] = (0.15, 0.85)
    # Items per gate per provider, chosen nearest to 0.5 within the band.
    n_select: int = 20


class VisionConfig(BaseModel):
    """Vision-arm configuration."""

    repeats: int = 6


class GateInferenceConfig(BaseModel):
    """Shared inference knobs."""

    temperature: Optional[float] = None
    max_retries: int = 6
    rate_limit_sleep: float = 1.5
    workers: int = 8
    seed: int = 42
    # Which gate prompt wording to use: "v1" (terse) or "rubric" (explicit
    # written criteria). Selects the GateSpec table in gates.prompts.
    prompt_variant: str = "v1"


class CustomGateConfig(BaseModel):
    """A user-defined gate: its decision prompt and stimulus bank.

    Lets a team run the boundary-calibrated drift methodology on their own
    gate without touching the built-in banks. The stimulus CSV must have
    columns: item_id, gate, scenario, level, text (levels form a graded
    weak->strong ladder so calibration can find the decision boundary).
    """

    system_prompt: str
    question: str
    yes_is_favorable: bool = True
    prompt_id: str = ""
    stimulus_path: Optional[str] = None


class GatesConfig(BaseModel):
    """Top-level configuration for a gates experiment."""

    run_tag: str = "gates_v1"
    seed: int = 42
    gates: List[str] = Field(default_factory=lambda: ["resume", "moderation", "appeal"])
    # name -> definition; names listed in `gates` resolve here first
    custom_gates: Dict[str, CustomGateConfig] = Field(default_factory=dict)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    variants: List[GateVariantSpec] = Field(default_factory=list)
    render: GateRenderConfig = Field(default_factory=GateRenderConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    inference: GateInferenceConfig = Field(default_factory=GateInferenceConfig)


def load_gates_config(path: str | Path) -> GatesConfig:
    """Load gates configuration from YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text()) or {}
    return GatesConfig(**data)
