from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from typography_eval import config
from typography_eval.src import fonts

IMG_WIDTH = 1200
MAX_TEXT_WIDTH = 1000
BG_COLOR = "white"
TEXT_COLOR = "black"
LINE_SPACING = 1.25

CONTAINER_PAD_X = 80
CONTAINER_PAD_Y = 40
CONTAINER_RADIUS = 24
CONTAINER_FILL = (245, 246, 248)
CONTAINER_OUTLINE = (220, 223, 228)
CONTAINER_OUTLINE_W = 3


def _require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Rendering requires Pillow to be installed") from exc


def wrap_text(text, font, max_width, draw):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def draw_rounded_rect(draw, xy, radius, fill, outline, width):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def render_sentence(sentence_id: int, text: str, variants: Dict[str, dict]) -> None:
    _require_pillow()
    from PIL import Image, ImageDraw, ImageFont

    sentence_dir = config.IMAGES_DIR / f"sentence_{sentence_id:03d}"
    sentence_dir.mkdir(parents=True, exist_ok=True)

    for variant_id, cfg in variants.items():
        out_path = sentence_dir / f"{variant_id}.png"
        if out_path.exists():
            continue

        if not cfg.get("font_path"):
            raise FileNotFoundError(f"Missing font path for variant {variant_id}")

        temp_img = Image.new("RGB", (IMG_WIDTH, 1200), BG_COLOR)
        draw = ImageDraw.Draw(temp_img)

        font = ImageFont.truetype(cfg["font_path"], cfg["size"])
        display_text = text.upper() if cfg["uppercase"] else text

        lines = wrap_text(display_text, font, MAX_TEXT_WIDTH, draw)

        line_bbox = draw.textbbox((0, 0), "Ag", font=font)
        base_line_height = line_bbox[3] - line_bbox[1]
        line_height = int(base_line_height * LINE_SPACING)

        max_line_w = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            max_line_w = max(max_line_w, bbox[2] - bbox[0])

        text_block_h = line_height * len(lines)

        container_w = min(IMG_WIDTH - 2 * CONTAINER_PAD_X, max_line_w + 2 * CONTAINER_PAD_X)
        container_h = text_block_h + 2 * CONTAINER_PAD_Y

        total_height = container_h + 80
        img = Image.new("RGB", (IMG_WIDTH, total_height), BG_COLOR)
        draw = ImageDraw.Draw(img)

        cx0 = (IMG_WIDTH - container_w) // 2
        cy0 = (total_height - container_h) // 2
        cx1 = cx0 + container_w
        cy1 = cy0 + container_h

        draw_rounded_rect(
            draw,
            (cx0, cy0, cx1, cy1),
            radius=CONTAINER_RADIUS,
            fill=CONTAINER_FILL,
            outline=CONTAINER_OUTLINE,
            width=CONTAINER_OUTLINE_W,
        )

        y = cy0 + (container_h - text_block_h) // 2
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            x = (IMG_WIDTH - w) // 2
            draw.text((x, y), line, fill=TEXT_COLOR, font=font)
            y += line_height

        img.save(out_path)


def render_all(sentences: Iterable[dict], variants: Dict[str, dict] | None = None) -> None:
    if variants is None:
        variants = fonts.typography_variants()
    for sentence in sentences:
        render_sentence(int(sentence["sentence_id"]), sentence["text"], variants)


def render_accessibility_only(sentences: Iterable[dict]) -> None:
    variants = fonts.typography_variants()
    access = {"A1_opendyslexic_regular": variants["A1_opendyslexic_regular"]}
    for sentence in sentences:
        render_sentence(int(sentence["sentence_id"]), sentence["text"], access)
