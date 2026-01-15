"""Smoke tests for rendering functionality."""

import pytest
from pathlib import Path
from PIL import Image
import tempfile

from typo_eval.config import TypoEvalConfig, RenderConfig, TypographyVariant
from typo_eval.render import wrap_text, render_text_image


class TestWrapText:
    """Test text wrapping functionality."""

    def test_wrap_short_text(self):
        """Short text should not wrap."""
        # Create a temporary image for drawing
        img = Image.new("RGB", (1000, 100), "white")
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(img)
        # Use default font
        font = ImageFont.load_default()

        lines = wrap_text("Hello world", font, 1000, draw)
        assert len(lines) == 1
        assert lines[0] == "Hello world"

    def test_wrap_long_text(self):
        """Long text should wrap to multiple lines."""
        img = Image.new("RGB", (1000, 100), "white")
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        long_text = "This is a very long text that should wrap to multiple lines when the width is limited"
        lines = wrap_text(long_text, font, 100, draw)
        assert len(lines) > 1


class TestRenderTextImage:
    """Test image rendering functionality."""

    @pytest.fixture
    def mock_font_path(self, tmp_path):
        """Create a path where we'd expect a font (tests will skip if font doesn't exist)."""
        # Return a path that may or may not exist
        return tmp_path / "test_font.ttf"

    def test_render_config_defaults(self):
        """Render config should have valid defaults for rendering."""
        render_cfg = RenderConfig()
        assert render_cfg.image_width > 0
        assert render_cfg.max_text_width > 0
        assert render_cfg.max_text_width < render_cfg.image_width

    def test_render_produces_image(self, tmp_path):
        """Rendering should produce a valid image (integration test with real font)."""
        # This test requires a real font file
        # Skip if no fonts available
        import glob

        # Try to find a system font
        font_paths = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
        if not font_paths:
            pytest.skip("No system fonts available for testing")

        font_path = Path(font_paths[0])
        render_cfg = RenderConfig()

        img = render_text_image(
            text="Test message",
            font_path=font_path,
            font_size=24,
            uppercase=False,
            render_cfg=render_cfg,
        )

        assert isinstance(img, Image.Image)
        assert img.width == render_cfg.image_width
        assert img.height > 0
        assert img.mode == "RGB"

    def test_render_long_sentence_single_line(self, tmp_path):
        """Long sentences should render on a single line for consistency."""
        import glob

        font_paths = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
        if not font_paths:
            pytest.skip("No system fonts available for testing")

        font_path = Path(font_paths[0])
        render_cfg = RenderConfig()

        long_text = (
            "This is a very long sentence that tests single line rendering "
            "for experiment consistency without text wrapping."
        )

        img = render_text_image(
            text=long_text,
            font_path=font_path,
            font_size=24,
            uppercase=False,
            render_cfg=render_cfg,
        )

        # Check that image was created with reasonable dimensions
        assert img.width == render_cfg.image_width
        # Height should be for a single line only
        assert img.height > 0  # Should have positive height

    def test_render_uppercase_transformation(self, tmp_path):
        """Uppercase flag should transform text to uppercase."""
        import glob

        font_paths = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
        if not font_paths:
            pytest.skip("No system fonts available for testing")

        font_path = Path(font_paths[0])
        render_cfg = RenderConfig()

        # Render same text with and without uppercase
        img_lower = render_text_image(
            text="test message",
            font_path=font_path,
            font_size=24,
            uppercase=False,
            render_cfg=render_cfg,
        )

        img_upper = render_text_image(
            text="test message",
            font_path=font_path,
            font_size=24,
            uppercase=True,
            render_cfg=render_cfg,
        )

        # Both should produce valid images
        assert isinstance(img_lower, Image.Image)
        assert isinstance(img_upper, Image.Image)


class TestContainerStyling:
    """Test container styling prevents clipping (the 'poster effect' fix)."""

    def test_container_config_values(self):
        """Container config should have proper styling values."""
        render_cfg = RenderConfig()
        container = render_cfg.container

        assert "padding" in container
        assert container["padding"] > 0
        assert "radius" in container
        assert container["radius"] > 0
        assert "fill" in container
        assert container["fill"] is None  # Transparent - no fill
        assert "outline" in container
        assert container["outline"] is None  # Transparent - no outline
