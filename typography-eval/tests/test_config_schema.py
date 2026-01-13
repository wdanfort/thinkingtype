"""Tests for configuration schema and loading."""

import pytest
from pathlib import Path

from typo_eval.config import (
    TypoEvalConfig,
    load_config,
    config_to_dict,
    RenderConfig,
    InferenceConfig,
    TypographyVariant,
)


class TestConfigSchema:
    """Test configuration schema validation."""

    def test_default_config_loads(self):
        """Default config should be valid and loadable."""
        config = TypoEvalConfig()
        assert config.seed == 42
        assert config.inference.temperature == 0.0
        assert config.inference.mode in ("dimensions", "decision", "both")

    def test_typography_variant_schema(self):
        """Typography variant should have required fields."""
        variant = TypographyVariant(
            id="test_variant",
            font_file="TestFont.ttf",
            size=48,
        )
        assert variant.id == "test_variant"
        assert variant.font_file == "TestFont.ttf"
        assert variant.size == 48
        assert variant.uppercase is False  # default
        assert variant.line_spacing == 1.25  # default

    def test_render_config_defaults(self):
        """Render config should have sensible defaults."""
        render_cfg = RenderConfig()
        assert render_cfg.image_width == 1200
        assert render_cfg.max_text_width == 1000
        assert "padding" in render_cfg.container
        assert "radius" in render_cfg.container

    def test_inference_config_defaults(self):
        """Inference config should have sensible defaults."""
        inference_cfg = InferenceConfig()
        assert inference_cfg.temperature == 0.0
        assert inference_cfg.mode == "both"
        assert len(inference_cfg.dimensions) == 10
        assert inference_cfg.decision_prompt_id == "escalation_v1"

    def test_config_to_dict_roundtrip(self):
        """Config should serialize to dict and back."""
        config = TypoEvalConfig()
        data = config_to_dict(config)
        assert isinstance(data, dict)
        assert "seed" in data
        assert "inference" in data
        assert data["inference"]["temperature"] == 0.0


class TestConfigLoading:
    """Test configuration file loading."""

    def test_load_config_file_not_found(self):
        """Loading non-existent config should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_load_v0_default_config(self, tmp_path):
        """V0 default config should be valid."""
        # Create a minimal config file
        config_content = """
seed: 123
inference:
  temperature: 0.5
  mode: dimensions
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)
        assert config.seed == 123
        assert config.inference.temperature == 0.5
        assert config.inference.mode == "dimensions"


class TestTypographyVariants:
    """Test typography variant configuration."""

    def test_variant_with_all_fields(self):
        """Variant with all fields specified."""
        variant = TypographyVariant(
            id="custom_variant",
            font_file="CustomFont.otf",
            size=36,
            weight="bold",
            uppercase=True,
            line_spacing=1.5,
        )
        assert variant.id == "custom_variant"
        assert variant.size == 36
        assert variant.uppercase is True
        assert variant.line_spacing == 1.5

    def test_config_with_variants(self):
        """Config with typography variants list."""
        config = TypoEvalConfig(
            typography_variants=[
                TypographyVariant(id="v1", font_file="Font1.ttf"),
                TypographyVariant(id="v2", font_file="Font2.ttf"),
            ]
        )
        assert len(config.typography_variants) == 2
        assert config.typography_variants[0].id == "v1"
        assert config.typography_variants[1].id == "v2"
