"""US Letter resume rendering for hrsim4.

Differences from the v1-v3 renderer, all of them audit fixes:

* **Real page geometry.** Pages are US Letter proportioned (8.5 x 11) at 1205 x
  1560 px, which keeps both dimensions under Anthropic's 1568 px downscale
  threshold so no arm is silently resampled.
* **Real resume layout.** Section headings with rules, job entries with
  right-aligned date columns, hanging-indent bullets, and two-column skill
  blocks -- instead of one wrapped prose blob.
* **No truncation.** Content flows onto as many pages as it needs and every
  page is sent. A resume set in OpenDyslexic legitimately runs longer than the
  same resume set in Liberation Sans; that is a real consequence of the font,
  so page count is recorded as an outcome rather than engineered away.
* **No bold anywhere.** Emphasis is carried by size, capitalisation, and rules,
  identical in every arm. Only three of the six typefaces ship a bold face, so
  using bold for headings would have given those arms real bold and left the
  others with none -- reintroducing exactly the ink-density confound the weight
  ladder exists to measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from typo_eval.hrsim4.parse import Bullets, Entry, Para, Resume, Section

# Letter proportions at ~142 dpi. Both axes stay below 1568 px.
PAGE_W = 1205
PAGE_H = 1560
DPI = PAGE_H / 11.0


def pt(points: float) -> int:
    """Typographic points -> pixels at the page's dpi."""
    return max(1, int(round(points * DPI / 72.0)))


@dataclass
class Style:
    """Point sizes and spacing. Identical across arms; only the face changes."""

    margin: int = pt(40)          # ~0.55 inch, typical for a dense resume
    body_pt: float = 9.0
    name_pt: float = 16.0
    headline_pt: float = 10.0
    contact_pt: float = 8.0
    heading_pt: float = 10.0
    line_spacing: float = 1.13
    space_section: int = pt(8)
    space_entry: int = pt(5)
    space_para: int = pt(2)
    bullet_indent: int = pt(12)
    rule_gap: int = pt(3)
    date_col_frac: float = 0.26


@dataclass
class Draw:
    """One positioned string, relative to its row's top-left."""

    x: int
    y: int
    text: str
    size: float


@dataclass
class Row:
    """An atomic unit of vertical flow.

    Paragraphs contribute one Row per wrapped line so they can break across
    pages; a two-column skill block contributes a single tall Row so its
    columns can never be split.
    """

    height: int
    draws: List[Draw] = field(default_factory=list)
    space_before: int = 0
    rule_after: bool = False
    keep_with_next: int = 0


class ResumeRenderer:
    def __init__(self, font_path: Path, base_scale: float = 1.0, style: Optional[Style] = None):
        self.font_path = Path(font_path)
        self.style = style or Style()
        self.base_scale = base_scale
        self._cache: dict = {}
        self._scratch = ImageDraw.Draw(Image.new("RGB", (10, 10)))

    # ------------------------------------------------------------- font utils
    def font(self, size_pt: float) -> ImageFont.FreeTypeFont:
        px = max(1, int(round(pt(size_pt) * self.base_scale)))
        key = px
        if key not in self._cache:
            self._cache[key] = ImageFont.truetype(str(self.font_path), px)
        return self._cache[key]

    def _width(self, text: str, size_pt: float) -> float:
        return self._scratch.textlength(text, font=self.font(size_pt))

    def _line_h(self, size_pt: float) -> int:
        f = self.font(size_pt)
        asc, desc = f.getmetrics()
        return int(round((asc + desc) * self.style.line_spacing))

    def _wrap(self, text: str, size_pt: float, max_w: float) -> List[str]:
        words = text.split()
        lines: List[str] = []
        cur = ""
        for w in words:
            cand = f"{cur} {w}" if cur else w
            if self._width(cand, size_pt) <= max_w or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines or [""]

    # ---------------------------------------------------------------- layout
    def _text_rows(
        self, text: str, size_pt: float, x: int, max_w: float,
        space_before: int = 0, hanging: int = 0, marker: str = "",
    ) -> List[Row]:
        lines = self._wrap(text, size_pt, max_w - (hanging if marker else 0))
        lh = self._line_h(size_pt)
        rows: List[Row] = []
        for i, ln in enumerate(lines):
            draws: List[Draw] = []
            if marker and i == 0:
                draws.append(Draw(x, 0, marker, size_pt))
            draws.append(Draw(x + (hanging if marker else 0), 0, ln, size_pt))
            rows.append(Row(height=lh, draws=draws, space_before=space_before if i == 0 else 0))
        return rows

    def _bullet_rows(self, b: Bullets, x: int, max_w: float) -> List[Row]:
        st = self.style
        columns = b.columns
        # A skills section of 30 one- or two-word bullets set one per line is
        # both unrealistic and enormously wasteful of page height -- it was the
        # main reason early revisions ran to three pages. Real resumes set
        # short lists in columns.
        if columns == 1 and len(b.items) >= 6:
            longest = max(len(i) for i in b.items)
            if longest <= 34:
                columns = 3
            elif longest <= 55:
                columns = 2

        if columns >= 2 and len(b.items) > 1:
            return self._column_rows(b.items, columns, x, max_w)

        rows: List[Row] = []
        for item in b.items:
            rows.extend(
                self._text_rows(
                    item, st.body_pt, x, max_w,
                    space_before=st.space_para, hanging=st.bullet_indent, marker="•",
                )
            )
        return rows

    def _column_rows(self, items: List[str], ncols: int, x: int, max_w: float) -> List[Row]:
        """Lay a short list into N balanced columns as one atomic Row."""
        st = self.style
        gutter = pt(10)
        col_w = (max_w - gutter * (ncols - 1)) / ncols
        lh = self._line_h(st.body_pt)
        per = (len(items) + ncols - 1) // ncols
        cols = [items[i * per:(i + 1) * per] for i in range(ncols)]
        draws: List[Draw] = []
        heights = []
        for ci, col in enumerate(cols):
            cx = int(x + ci * (col_w + gutter))
            y = 0
            for item in col:
                for i, ln in enumerate(
                    self._wrap(item, st.body_pt, col_w - st.bullet_indent)
                ):
                    if i == 0:
                        draws.append(Draw(cx, y, "•", st.body_pt))
                    draws.append(Draw(cx + st.bullet_indent, y, ln, st.body_pt))
                    y += lh
            heights.append(y)
        return [Row(height=max(heights) if heights else 0, draws=draws,
                    space_before=st.space_para)]

    def _entry_rows(self, e: Entry, x: int, max_w: float) -> List[Row]:
        st = self.style
        rows: List[Row] = []
        date_w = max_w * st.date_col_frac
        head_w = max_w - date_w - pt(8)

        head = " — ".join(p for p in (e.title, e.org) if p)
        if head or e.dates:
            lh = self._line_h(st.body_pt)
            lines = self._wrap(head, st.body_pt, head_w) if head else [""]
            for i, ln in enumerate(lines):
                draws = [Draw(x, 0, ln, st.body_pt)]
                if i == 0 and e.dates:
                    dw = self._width(e.dates, st.body_pt)
                    draws.append(Draw(int(x + max_w - dw), 0, e.dates, st.body_pt))
                rows.append(
                    Row(height=lh, draws=draws,
                        space_before=st.space_entry if i == 0 else 0,
                        keep_with_next=2 if i == 0 else 0)
                )
        if e.location:
            rows.extend(self._text_rows(e.location, st.contact_pt, x, max_w))

        for b in e.body:
            if isinstance(b, Bullets):
                rows.extend(self._bullet_rows(b, x + st.bullet_indent, max_w - st.bullet_indent))
            else:
                rows.extend(
                    self._text_rows(b.text, st.body_pt, x, max_w, space_before=st.space_para)
                )
        return rows

    def _section_rows(self, sec: Section, x: int, max_w: float) -> List[Row]:
        st = self.style
        rows = self._text_rows(sec.title.upper(), st.heading_pt, x, max_w,
                               space_before=st.space_section)
        rows[-1].rule_after = True
        rows[-1].keep_with_next = 3
        for item in sec.items:
            if isinstance(item, Entry):
                rows.extend(self._entry_rows(item, x, max_w))
            elif isinstance(item, Bullets):
                rows.extend(self._bullet_rows(item, x, max_w))
            else:
                rows.extend(
                    self._text_rows(item.text, st.body_pt, x, max_w, space_before=st.space_para)
                )
        return rows

    def layout(self, resume: Resume) -> List[Row]:
        st = self.style
        x = st.margin
        max_w = PAGE_W - 2 * st.margin
        rows: List[Row] = []
        rows.extend(self._text_rows(resume.name, st.name_pt, x, max_w))
        if resume.headline:
            rows.extend(self._text_rows(resume.headline, st.headline_pt, x, max_w,
                                        space_before=pt(2)))
        if resume.contact:
            rows.extend(self._text_rows(" | ".join(resume.contact), st.contact_pt, x, max_w,
                                        space_before=pt(2)))
        for sec in resume.sections:
            rows.extend(self._section_rows(sec, x, max_w))
        return rows

    # ------------------------------------------------------------ pagination
    def paginate(self, rows: Sequence[Row]) -> List[List[Tuple[Row, int]]]:
        st = self.style
        usable = PAGE_H - st.margin
        pages: List[List[Tuple[Row, int]]] = []
        cur: List[Tuple[Row, int]] = []
        y = st.margin
        i = 0
        while i < len(rows):
            r = rows[i]
            top = y + (r.space_before if cur else 0)
            need = r.height + (st.rule_gap * 2 if r.rule_after else 0)
            # A heading must not be stranded at the foot of a page.
            if r.keep_with_next:
                lookahead = need
                for k in range(1, r.keep_with_next + 1):
                    if i + k < len(rows):
                        lookahead += rows[i + k].space_before + rows[i + k].height
                fits = top + lookahead <= usable
            else:
                fits = top + need <= usable

            if not fits and cur:
                pages.append(cur)
                cur = []
                y = st.margin
                continue
            cur.append((r, top))
            y = top + need
            i += 1
        if cur:
            pages.append(cur)
        return pages

    # ---------------------------------------------------------------- render
    def render(self, resume: Resume) -> List[Image.Image]:
        st = self.style
        pages = self.paginate(self.layout(resume))
        images: List[Image.Image] = []
        for page in pages:
            img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
            d = ImageDraw.Draw(img)
            for row, top in page:
                for dr in row.draws:
                    d.text((dr.x, top + dr.y), dr.text, font=self.font(dr.size), fill="black")
                if row.rule_after:
                    ry = top + row.height + st.rule_gap
                    d.line(
                        [(st.margin, ry), (PAGE_W - st.margin, ry)],
                        fill=(120, 120, 120), width=1,
                    )
            images.append(img)
        return images
