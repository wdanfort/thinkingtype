"""Configuration loader and schema."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class RenderingConfig(BaseModel):
    canvas_width: int = 1200
    max_text_width: int = 1000
    background_color: str = "white"
    text_color: str = "black"
    line_spacing: float = 1.25
    container_pad_x: int = 80
    container_pad_y: int = 40
    container_radius: int = 24
    container_fill: List[int] = Field(default_factory=lambda: [245, 246, 248])
    container_outline: List[int] = Field(default_factory=lambda: [220, 223, 228])
    container_outline_width: int = 3
    title_multiplier: float = 1.0


class InferenceConfig(BaseModel):
    provider_text: str = "openai"
    model_text: str = "gpt-4o-mini"
    provider_image: str = "openai"
    model_image: str = "gpt-4o"
    temperature: float = 0.0
    max_retries: int = 6
    timeout: int = 30
    rate_limit_sleep: float = 1.5
    fail_fast: bool = False


class OutputConfig(BaseModel):
    runs_root: str = "runs"
    artifacts_root: str = "artifacts"


class TypographyConfig(BaseModel):
    artifact_set_id: str = "sentences_v0"
    sentences_csv: str = "data/sentences_v0.csv"
    variants_csv: str = "data/variants_v0.csv"
    output_dir: str = "artifacts/sentences_v0"
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


DEFAULT_CONFIG_PATH = Path("configs/v0.yaml")


def load_config(path: Optional[str] = None) -> TypographyConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text()) or {}
    return TypographyConfig(**data)


def config_to_dict(config: TypographyConfig) -> dict:
    return config.model_dump()


def resolve_paths(config: TypographyConfig, root: Optional[Path] = None) -> TypographyConfig:
    base = root or Path.cwd()
    resolved = config.model_copy(deep=True)
    resolved.sentences_csv = str((base / resolved.sentences_csv).resolve())
    resolved.variants_csv = str((base / resolved.variants_csv).resolve())
    resolved.output_dir = str((base / resolved.output_dir).resolve())
    resolved.output.runs_root = str((base / resolved.output.runs_root).resolve())
    resolved.output.artifacts_root = str((base / resolved.output.artifacts_root).resolve())
    return resolved
