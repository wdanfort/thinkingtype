"""Realistic app/forum "chrome" rendering for gate documents.

Renders the same document text used by ``render.render_gate_document`` but
wrapped in a plausible screenshot chrome (avatar, handle, timestamp, action
row, and for the phone variant a status bar) so that a vision-language model
sees a "real" screenshot rather than a plain text block. This lets us test
whether chrome framing shifts judgments relative to the plain-block render of
the identical text.

All handles are synthetic, demographically-neutral, and deterministic given
the document text (same text -> same handle/timestamp/count within a single
process run), so re-rendering a document is stable.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

from typo_eval.gates.config import GateRenderConfig, GateVariantSpec
from typo_eval.gates.render import _wrap_paragraph

PHONE_WIDTH = 720


def _stable_hash(text: str) -> int:
    """Process-independent hash (built-in hash() is randomized per process,
    which would change handles/timestamps between render runs)."""
    import hashlib

    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def _neutral_handle(text: str) -> str:
    """Deterministic, demographically-neutral synthetic handle."""
    return "member_" + str(_stable_hash(text) % 9000 + 1000)


def _neutral_timestamp(text: str) -> str:
    """Deterministic relative timestamp, e.g. '3h'."""
    hours = (_stable_hash("ts:" + text) % 12) + 1
    return f"{hours}h"


def _neutral_upvotes(text: str) -> int:
    """Deterministic small upvote count."""
    return (_stable_hash("uv:" + text) % 500) + 1


def _theme_colors(chrome: str) -> dict:
    """Fixed color palette per chrome style (not derived from variant)."""
    if chrome == "forum_dark":
        return dict(
            page_bg="#18191A",
            card_bg="#242526",
            card_border="#3A3B3C",
            text_color="#E4E6EB",
            muted_color="#B0B3B8",
            avatar_color="#4E4F50",
            divider_color="#3A3B3C",
        )
    # forum_light and phone_light share the same light palette
    return dict(
        page_bg="#F0F2F5",
        card_bg="#FFFFFF",
        card_border="#D8DEE4",
        text_color="#1C1E21",
        muted_color="#65676B",
        avatar_color="#B0B3B8",
        divider_color="#E4E6EB",
    )


def _label_font(font_path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path), max(10, size))


def _line_height_for(
    font: ImageFont.FreeTypeFont, draw: ImageDraw.ImageDraw, spacing: float = 1.0
) -> int:
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    return int((bbox[3] - bbox[1]) * spacing)


def render_chrome_document(
    text: str,
    variant: GateVariantSpec,
    font_path: Path,
    font_size: int,
    render_cfg: GateRenderConfig,
) -> Image.Image:
    """Render a gate document inside realistic forum/phone screenshot chrome.

    The chrome style is read from ``variant.chrome`` (a field the base
    ``GateVariantSpec`` schema does not declare, so it is accessed via
    ``getattr`` and defaults to "forum_light" if unset/unknown).
    """
    chrome = getattr(variant, "chrome", None) or "forum_light"
    if chrome not in ("forum_light", "forum_dark", "phone_light"):
        chrome = "forum_light"

    colors = _theme_colors(chrome)
    is_phone = chrome == "phone_light"

    display_text = text.upper() if variant.uppercase else text

    body_font = ImageFont.truetype(str(font_path), font_size)
    scale = max(0.5, min(2.0, font_size / 40.0))

    label_size = max(12, int(font_size * 0.5))
    small_size = max(11, int(font_size * 0.42))
    label_font = _label_font(font_path, label_size)
    small_font = _label_font(font_path, small_size)

    image_width = PHONE_WIDTH if is_phone else render_cfg.image_width
    outer_margin = int((16 if is_phone else 40) * scale)
    card_padding = int((24 if is_phone else 36) * scale)
    avatar_diameter = int((48 if is_phone else 64) * scale)
    status_bar_height = int(44 * scale) if is_phone else 0

    card_width = image_width - 2 * outer_margin
    text_max_width = min(
        render_cfg.max_text_width,
        card_width - 2 * card_padding,
    )

    # Scratch canvas for measurement, matching render.py's approach.
    scratch = Image.new("RGB", (image_width, 200), colors["page_bg"])
    sdraw = ImageDraw.Draw(scratch)

    base_line_height = _line_height_for(body_font, sdraw)
    body_line_height = int(base_line_height * variant.line_spacing)

    lines: List[str] = []
    for paragraph in display_text.split("\n"):
        if paragraph.strip():
            lines.extend(_wrap_paragraph(paragraph, body_font, text_max_width, sdraw))
        else:
            lines.append("")
    body_height = len(lines) * body_line_height

    label_line_height = _line_height_for(label_font, sdraw)
    small_line_height = _line_height_for(small_font, sdraw)

    header_height = max(avatar_diameter, label_line_height + small_line_height + 4)
    header_body_gap = int(20 * scale)
    body_footer_gap = int(24 * scale)
    footer_height = max(small_line_height, int(28 * scale))

    card_height = (
        2 * card_padding
        + header_height
        + header_body_gap
        + body_height
        + body_footer_gap
        + footer_height
    )

    total_height = (
        outer_margin + status_bar_height + card_height + outer_margin
    )

    img = Image.new("RGB", (image_width, total_height), colors["page_bg"])
    draw = ImageDraw.Draw(img)

    # --- Phone status bar -------------------------------------------------
    if is_phone:
        sb_text_y = (status_bar_height - small_line_height) // 2
        draw.text(
            (outer_margin, sb_text_y),
            "9:41",
            fill=colors["text_color"],
            font=small_font,
        )
        # Signal bars (ascending heights), wifi dot, battery outline.
        bar_x = image_width - outer_margin - int(90 * scale)
        bar_w = int(4 * scale)
        bar_gap = int(3 * scale)
        max_bar_h = int(10 * scale)
        for i in range(4):
            bh = int(max_bar_h * (i + 1) / 4)
            bx0 = bar_x + i * (bar_w + bar_gap)
            by1 = sb_text_y + small_line_height - 1
            by0 = by1 - bh
            draw.rectangle(
                (bx0, by0, bx0 + bar_w, by1), fill=colors["text_color"]
            )
        wifi_cx = bar_x + int(45 * scale)
        wifi_r = int(5 * scale)
        wifi_cy = sb_text_y + small_line_height // 2
        draw.ellipse(
            (
                wifi_cx - wifi_r,
                wifi_cy - wifi_r,
                wifi_cx + wifi_r,
                wifi_cy + wifi_r,
            ),
            outline=colors["text_color"],
            width=max(1, int(1 * scale)),
        )
        batt_x0 = bar_x + int(60 * scale)
        batt_w = int(22 * scale)
        batt_h = int(11 * scale)
        batt_y0 = sb_text_y + (small_line_height - batt_h) // 2
        draw.rounded_rectangle(
            (batt_x0, batt_y0, batt_x0 + batt_w, batt_y0 + batt_h),
            radius=max(1, int(2 * scale)),
            outline=colors["text_color"],
            width=max(1, int(1 * scale)),
        )
        fill_pad = max(1, int(2 * scale))
        draw.rectangle(
            (
                batt_x0 + fill_pad,
                batt_y0 + fill_pad,
                batt_x0 + int(batt_w * 0.7),
                batt_y0 + batt_h - fill_pad,
            ),
            fill=colors["text_color"],
        )

    # --- Card ---------------------------------------------------------------
    card_top = outer_margin + status_bar_height
    card_left = outer_margin
    card_right = image_width - outer_margin
    card_bottom = card_top + card_height
    radius = int(16 * scale)
    draw.rounded_rectangle(
        (card_left, card_top, card_right, card_bottom),
        radius=radius,
        fill=colors["card_bg"],
        outline=colors["card_border"],
        width=max(1, int(1 * scale)),
    )

    # --- Header: avatar, handle, timestamp -----------------------------------
    avatar_x0 = card_left + card_padding
    avatar_y0 = card_top + card_padding
    avatar_x1 = avatar_x0 + avatar_diameter
    avatar_y1 = avatar_y0 + avatar_diameter
    draw.ellipse(
        (avatar_x0, avatar_y0, avatar_x1, avatar_y1), fill=colors["avatar_color"]
    )

    handle = _neutral_handle(text)
    timestamp = _neutral_timestamp(text)

    header_text_x = avatar_x1 + int(16 * scale)
    header_text_y = avatar_y0 + (avatar_diameter - (label_line_height + small_line_height)) // 2
    draw.text(
        (header_text_x, header_text_y),
        handle,
        fill=colors["text_color"],
        font=label_font,
    )
    draw.text(
        (header_text_x, header_text_y + label_line_height + 2),
        timestamp,
        fill=colors["muted_color"],
        font=small_font,
    )

    # --- Body: the document text, verbatim, wrapped --------------------------
    body_y = card_top + card_padding + header_height + header_body_gap
    body_x = card_left + card_padding
    for line in lines:
        if line:
            if variant.align == "center":
                bbox = draw.textbbox((0, 0), line, font=body_font)
                line_width = bbox[2] - bbox[0]
                x = card_left + (card_width - line_width) // 2
            else:
                x = body_x
            if variant.highlight_color:
                bbox = draw.textbbox((0, 0), line, font=body_font)
                line_width = bbox[2] - bbox[0]
                pad = max(2, base_line_height // 8)
                draw.rectangle(
                    (
                        x - pad,
                        body_y - pad,
                        x + line_width + pad,
                        body_y + base_line_height + pad,
                    ),
                    fill=variant.highlight_color,
                )
            draw.text((x, body_y), line, fill=colors["text_color"], font=body_font)
        body_y += body_line_height

    # --- Divider --------------------------------------------------------------
    divider_y = card_top + card_padding + header_height + header_body_gap + body_height + (
        body_footer_gap // 2
    )
    draw.line(
        (card_left + card_padding, divider_y, card_right - card_padding, divider_y),
        fill=colors["divider_color"],
        width=max(1, int(1 * scale)),
    )

    # --- Footer: upvote arrow + count, then muted action affordances ---------
    footer_y = card_top + card_height - card_padding - footer_height
    upvotes = _neutral_upvotes(text)

    arrow_size = int(footer_height * 0.5)
    arrow_x = card_left + card_padding
    arrow_y_bottom = footer_y + footer_height
    arrow_y_top = arrow_y_bottom - arrow_size
    draw.polygon(
        [
            (arrow_x + arrow_size // 2, arrow_y_top),
            (arrow_x, arrow_y_bottom),
            (arrow_x + arrow_size, arrow_y_bottom),
        ],
        fill=colors["muted_color"],
    )
    count_x = arrow_x + arrow_size + int(8 * scale)
    draw.text(
        (count_x, footer_y + (footer_height - small_line_height) // 2),
        str(upvotes),
        fill=colors["muted_color"],
        font=small_font,
    )

    actions_text = "Reply      Report      Share"
    actions_x = count_x + int(70 * scale)
    draw.text(
        (actions_x, footer_y + (footer_height - small_line_height) // 2),
        actions_text,
        fill=colors["muted_color"],
        font=small_font,
    )

    return img


if __name__ == "__main__":
    from typo_eval.gates.config import GateRenderConfig, GateVariantSpec

    class _ChromeVariantSpec(GateVariantSpec):
        """Smoke-test-only subclass adding the `chrome` field.

        The real experiment config is expected to extend GateVariantSpec
        with a `chrome: Optional[str] = None` field in config.py; this local
        subclass lets this module be smoke-tested standalone without
        touching any other file.
        """

        chrome: str = "forum_light"

    sample_text = (
        "APPEAL #4471\n"
        "Original decision: content removed for policy violation.\n"
        "Requester statement: I believe this removal was made in error. "
        "The post did not contain the material described in the notice, "
        "and I have attached the original file for review. Please "
        "reconsider this decision at your earliest convenience.\n"
        "Status: pending review."
    )

    render_cfg = GateRenderConfig()
    font_path = Path(
        "/System/Library/Fonts/Supplemental/Arial.ttf"
    )
    if not font_path.exists():
        # Fall back to a font that should exist inside the repo's assets.
        repo_root = Path(__file__).resolve().parents[3]
        candidates = list(repo_root.rglob("LiberationSans-Regular.ttf"))
        if candidates:
            font_path = candidates[0]

    font_size = 32

    for chrome_style in ("forum_light", "forum_dark", "phone_light"):
        variant = _ChromeVariantSpec(
            id=f"chrome_{chrome_style}",
            text_color="#000000",
            bg_color="#FFFFFF",
            chrome=chrome_style,
        )

        img = render_chrome_document(sample_text, variant, font_path, font_size, render_cfg)
        out_path = Path(f"/tmp/chrome_{chrome_style}.png")
        img.save(out_path)
        print(f"{chrome_style}: saved {out_path} size={img.size}")
