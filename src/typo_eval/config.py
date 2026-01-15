"""Configuration loader and schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class InputsConfig(BaseModel):
    """Configuration for input generation."""

    sentences: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "n_sentences": 36,
        "categories": ["neutral", "cta", "authority", "warning", "promo", "procedural"],
        "source": "synthetic",
        "path": None,
    })
    artifacts: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": False,
        "n_each_type": 6,
        "types": ["email", "notification", "alert", "form"],
        "source": "synthetic",
        "path": None,
    })


class TypographyVariant(BaseModel):
    """A typography variant configuration."""

    id: str
    font_file: str
    size: int = 48
    weight: str = "regular"
    uppercase: bool = False
    line_spacing: float = 1.25


class RenderConfig(BaseModel):
    """Rendering configuration."""

    image_width: int = 1200
    container: Dict[str, Any] = Field(default_factory=lambda: {
        "padding": 80,
        "radius": 24,
        "fill": [245, 246, 248],
        "outline": [220, 223, 228],
    })
    margins: int = 40
    max_text_width: int = 1000


class OCRConfig(BaseModel):
    """OCR configuration."""

    enabled: bool = True
    tesseract_lang: str = "eng"


class ProviderConfig(BaseModel):
    """Provider-specific configuration."""

    api_key_env: str = ""
    model_text: str = ""
    model_vision: str = ""


class ProvidersConfig(BaseModel):
    """All providers configuration."""

    openai: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        api_key_env="OPENAI_API_KEY",
        model_text="gpt-4o-mini",
        model_vision="gpt-4o",
    ))
    anthropic: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        api_key_env="ANTHROPIC_API_KEY",
        model_text="claude-sonnet-4-20250514",
        model_vision="claude-sonnet-4-20250514",
    ))
    google: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        api_key_env="GOOGLE_API_KEY",
        model_text="gemini-1.5-pro",
        model_vision="gemini-1.5-pro",
    ))


class InferenceConfig(BaseModel):
    """Inference configuration."""

    mode: str = "both"  # "dimensions" | "decision" | "both"
    temperature: float = 0.0
    repeats: int = 1
    dimensions: List[str] = Field(default_factory=lambda: [
        "urgent",
        "immediate_action",
        "formal",
        "trustworthy",
        "persuasive",
        "emotional",
        "professional",
        "high_risk",
        "confident",
        "form_dependent",
    ])
    decision_prompt_id: str = "escalation_v1"
    max_retries: int = 6
    rate_limit_sleep: float = 1.5
    fail_fast: bool = False


class BootstrapConfig(BaseModel):
    """Bootstrap CI configuration."""

    n_boot: int = 2000
    alpha: float = 0.05


class AnalysisConfig(BaseModel):
    """Analysis configuration."""

    compute_heatmaps: bool = True
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    outputs: Dict[str, bool] = Field(default_factory=lambda: {
        "save_csv": True,
        "save_png": True,
        "save_md": True,
    })


class TypoEvalConfig(BaseModel):
    """Main configuration schema."""

    run_id: Optional[str] = None
    seed: int = 42
    inputs: InputsConfig = Field(default_factory=InputsConfig)
    typography_variants: List[TypographyVariant] = Field(default_factory=list)
    render: RenderConfig = Field(default_factory=RenderConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


DEFAULT_CONFIG_PATH = Path("configs/v0_default.yaml")


def load_config(path: Optional[str | Path] = None) -> TypoEvalConfig:
    """Load configuration from YAML file."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text()) or {}
    return TypoEvalConfig(**data)


def config_to_dict(config: TypoEvalConfig) -> dict:
    """Convert config to dictionary."""
    return config.model_dump()


def get_repo_root() -> Path:
    """Get repository root path."""
    candidate = Path.cwd()
    for p in [candidate, *candidate.parents]:
        if (p / "configs").exists() and (p / "assets").exists():
            return p
    return candidate


def resolve_font_path(font_file: str, repo_root: Optional[Path] = None) -> Path:
    """Resolve font file path relative to repo root."""
    root = repo_root or get_repo_root()
    return (root / "assets" / "fonts" / font_file).resolve()
