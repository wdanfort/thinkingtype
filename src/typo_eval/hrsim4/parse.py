"""LiveCareer `Resume_html` -> structured resume document.

The v1-v3 corpus used the `Resume_str` column, which is the HTML with all tags
stripped: section headings collide with body text ("SPECIALISTNETWORK",
"PROFESSIONALEXPERIENCE. Apollo"), bullets become run-on sentences, and job
metadata (company / title / dates / location) loses its association with the
job it describes. The `Resume_html` column of the same CSV has all of it, with
semantic class names -- `sectiontitle`, `companyname`, `jobtitle`, `jobdates`,
`joblocation`, `degree`, `twocol` -- so the structure is recoverable
deterministically, with no LLM rewriting of candidate content.

Everything here is pure parsing plus placeholder substitution. Nothing in this
module varies by font: the parsed document is built once per resume and every
typographic arm renders the identical Resume object, so content can never be a
confound between arms.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Union

from bs4 import BeautifulSoup, Tag

# --------------------------------------------------------------------- model


@dataclass
class Para:
    """A run of body prose."""

    text: str


@dataclass
class Bullets:
    """A bullet list. `columns` mirrors LiveCareer's two-column skill tables."""

    items: List[str]
    columns: int = 1


Block = Union[Para, Bullets]


@dataclass
class Entry:
    """One job or degree: metadata line(s) plus its own body blocks."""

    title: str = ""
    org: str = ""
    location: str = ""
    dates: str = ""
    body: List[Block] = field(default_factory=list)


SectionItem = Union[Para, Bullets, Entry]


@dataclass
class Section:
    title: str
    items: List[SectionItem] = field(default_factory=list)


@dataclass
class Resume:
    resume_id: int
    category: str
    name: str = ""
    headline: str = ""
    contact: List[str] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)

    def plain_text(self) -> str:
        """Flatten to the text the text-arm sees.

        Must contain exactly the words the rendered image contains -- the text
        arm and the image arms have to be content-identical or the modality
        comparison is confounded. Structure is conveyed with newlines and "- "
        markers rather than being thrown away as it was in v1-v3.
        """
        out: List[str] = [self.name]
        if self.headline:
            out.append(self.headline)
        if self.contact:
            out.append(" | ".join(self.contact))
        for sec in self.sections:
            out.append("")
            out.append(sec.title.upper())
            for item in sec.items:
                out.extend(_item_lines(item))
        return "\n".join(out).strip()


def _block_lines(b: Block) -> List[str]:
    if isinstance(b, Para):
        return [b.text]
    return [f"- {i}" for i in b.items]


def _item_lines(item: SectionItem) -> List[str]:
    if isinstance(item, Entry):
        head = " | ".join(x for x in [item.title, item.org, item.location, item.dates] if x)
        lines = [head] if head else []
        for b in item.body:
            lines.extend(_block_lines(b))
        return lines
    return _block_lines(item)


# ------------------------------------------------------------ placeholder fill
# LiveCareer scrubbed employers to the literal string "Company Name" (404
# occurrences across 175 of v2's 180 resumes) and locations to "City , State".
# Left in, they make every resume read as an unfilled template. Filled in, they
# must not smuggle in a new signal, so the pools are deliberately bland
# invented firms with no real-world prestige, and the SAME pool is used for
# every occupation so employer identity cannot correlate with occupation or
# with the screening outcome. Substitution is seeded on resume_id, so a given
# resume gets the same employers in every arm and on every rebuild.

_COMPANY_HEADS = [
    "Ardent", "Beacon", "Calder", "Dunmore", "Eastfield", "Fairmount", "Granby",
    "Harlow", "Ivestone", "Jarrow", "Kelvin", "Larkspur", "Merrick", "Norwood",
    "Oakbridge", "Pemberton", "Quarry", "Redhill", "Stannard", "Thornbury",
    "Ulster", "Vantage", "Westmere", "Yardley",
]
_COMPANY_TAILS = [
    "Group", "Partners", "Industries", "Systems", "Associates", "Holdings",
    "Services", "Solutions", "Corporation", "Enterprises",
]
_CITIES = [
    ("Riverton", "OH"), ("Fairhaven", "MI"), ("Lakeside", "PA"), ("Clayton", "MO"),
    ("Brookfield", "WI"), ("Ashford", "NC"), ("Milford", "CT"), ("Danville", "IN"),
    ("Whitfield", "GA"), ("Kingsley", "IA"), ("Grantham", "NH"), ("Selby", "OR"),
]


def _rng_ints(seed_text: str, n: int) -> List[int]:
    """Deterministic integer stream from a seed string (stable across runs)."""
    out: List[int] = []
    counter = 0
    while len(out) < n:
        h = hashlib.sha256(f"{seed_text}:{counter}".encode()).digest()
        out.extend(h)
        counter += 1
    return out[:n]


class PlaceholderFiller:
    """Assigns stable fake employers/locations within one resume.

    Repeated placeholders inside a resume must map consistently: the same job
    entry should not be "Ardent Group" in its heading and "Beacon Partners" in
    its description. Assignment is by order of first appearance.
    """

    def __init__(self, resume_id: int) -> None:
        self._ints = _rng_ints(f"hrsim4:{resume_id}", 512)
        self._i = 0
        self._companies: List[str] = []
        self._cities: List[str] = []
        self._company_used = 0
        self._city_used = 0

    def _next(self) -> int:
        v = self._ints[self._i % len(self._ints)]
        self._i += 1
        return v

    def company(self, index: int) -> str:
        while len(self._companies) <= index:
            head = _COMPANY_HEADS[self._next() % len(_COMPANY_HEADS)]
            tail = _COMPANY_TAILS[self._next() % len(_COMPANY_TAILS)]
            candidate = f"{head} {tail}"
            if candidate in self._companies:
                candidate = f"{head} {_COMPANY_TAILS[(self._next()) % len(_COMPANY_TAILS)]}"
            self._companies.append(candidate)
        return self._companies[index]

    def city_state(self, index: int) -> str:
        while len(self._cities) <= index:
            city, state = _CITIES[self._next() % len(_CITIES)]
            self._cities.append(f"{city}, {state}")
        return self._cities[index]


# ------------------------------------------------------------------- parsing

# Characters present in the LiveCareer source that are NOT covered by every
# arm's font. Left alone they are a genuine cross-arm content confound: the
# same resume shows "Applied Science Engineering －" in one typeface and
# "Applied Science Engineering □" in another, because the fallback for a
# missing glyph is a tofu box. Everything here is folded to ASCII before
# layout, and `qc.audit` independently verifies that no character survives
# that any arm's font cannot draw.
_UNICODE_FOLD = {
    "－": "-", "−": "-", "–": "-", "—": "-", "‒": "-",
    "•": "", "·": "", "●": "", "○": "", "▪": "",
    "➢": "", "❏": "", "■": "", "‣": "", "⁃": "",
    "​": "", "­": "", "﻿": "", "�": "",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'",
    "…": "...", "™": "(TM)", "®": "(R)", "©": "(C)",
    "‹": "<", "›": ">", "«": '"', "»": '"',
    "→": "->", "⇒": "=>", "≤": "<=", "≥": ">=",
    "×": "x", "⁄": "/", "′": "'", "″": '"',
    # Comic Neue has no Greek. A corpus-wide scan against the intersection of
    # all eight arm fonts' cmaps found this to be the only letter any arm
    # cannot draw; "u" is the conventional transliteration ("ug" for micrograms).
    "μ": "u", "µ": "u",
}
_FOLD_RE = re.compile("|".join(re.escape(k) for k in _UNICODE_FOLD))


def _is_unrenderable(ch: str) -> bool:
    """True for symbol classes no text font reliably covers.

    The explicit table above handles the characters actually seen in this
    corpus, but resumes exported from Word carry arbitrary Wingdings bullets in
    the private-use area (U+F0A7 and friends) and assorted dingbats. Rather
    than extend the table each time a new one surfaces, drop whole categories:
    Co (private use) and So (other symbol). Letters, digits, punctuation and
    marks are never touched.
    """
    return unicodedata.category(ch) in {"Co", "So", "Cs", "Cn"}


def fold_unicode(text: str) -> str:
    """Map source typography to characters every arm's font can render."""
    text = unicodedata.normalize("NFKC", text)
    text = _FOLD_RE.sub(lambda m: _UNICODE_FOLD[m.group(0)], text)
    return "".join(" " if _is_unrenderable(c) else c for c in text)


_WS = re.compile(r"\s+")
_PLACEHOLDER_COMPANY = re.compile(r"\bCompany Name\b")
_PLACEHOLDER_CITY = re.compile(r"\bCity\s*,\s*State\b")
_SECTION_STOPWORDS = {"", "skills", "additional information"}


def _clean(text: str) -> str:
    return _WS.sub(" ", fold_unicode(text)).strip()


def _text_of(tag: Optional[Tag]) -> str:
    return _clean(tag.get_text(" ")) if tag is not None else ""


def _has_class(tag: Tag, name: str) -> bool:
    return name in (tag.get("class") or [])


_SEPARATORS = re.compile(r"^[\s,;:.\-–—/|]+|[\s,;:\-–—/|]+$")
_REPEAT_PUNCT = re.compile(r"(?:\s*[,;:]\s*){2,}")
_ORPHAN_TO = re.compile(r"^(?:to|and|of|at|in)\b[\s,;:.\-]*", re.I)


def _tidy(text: str) -> str:
    """Strip the debris left behind after metadata spans are decomposed.

    Removing a `.jobdates` span leaves its sibling separator text ("02/2011"
    and "Current" go, the literal " to " stays), and removing `.companyname`
    inside a comma-delimited line leaves doubled commas. Without this the body
    of every job entry starts ", to ".
    """
    text = _REPEAT_PUNCT.sub(", ", text)
    text = _SEPARATORS.sub("", text)
    text = _ORPHAN_TO.sub("", text)
    return _SEPARATORS.sub("", text)


def _is_substantive(text: str) -> bool:
    """True if a fragment carries content rather than leftover punctuation."""
    return sum(c.isalnum() for c in text) >= 3


def _dedupe_runs(text: str) -> str:
    """Collapse an immediately repeated phrase.

    LiveCareer education entries emit the degree twice (once in `.degree`,
    once in `.programline`), which after tag removal reads
    "Electronic Engineering , , Electronic Engineering".
    """
    def _key(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    for sep in (", ", " "):
        parts = [p for p in text.split(sep) if p]
        out: List[str] = []
        for p in parts:
            # Compare on alphanumerics only: the duplicate arrives as
            # "Applied Science Engineering -" vs "Applied Science Engineering",
            # which differ by the separator debris alone.
            if not out or not _key(p) or _key(out[-1]) != _key(p):
                out.append(p)
        text = sep.join(out)
    return text


def _split_sentencey(text: str) -> List[str]:
    """Split a run-on description into paragraph-sized chunks.

    LiveCareer descriptions arrive as one long <div> with sentences separated
    only by whitespace. A single 1,500-character paragraph is not what a real
    resume looks like, but neither is a stack of one-sentence fragments -- and
    over-splitting is expensive on the page, because every fragment wastes the
    tail of its last line. Splits on sentence boundaries and regroups to <= 600
    chars, which keeps summary paragraphs intact while breaking up the worst
    run-ons.
    """
    parts = re.split(r"(?<=[.!?])\s+", text)
    out: List[str] = []
    cur = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if cur and len(cur) + len(p) + 1 > 600:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        out.append(cur)
    return out


def _parse_blocks(node: Tag, filler: PlaceholderFiller, counters: dict) -> List[Block]:
    """Body content of a paragraph/entry: bullet lists and prose."""
    blocks: List[Block] = []

    for table in node.find_all("table"):
        items = [_tidy(_clean(p.get_text(" "))) for p in table.find_all("p")]
        items = [i for i in items if _is_substantive(i)]
        if items:
            blocks.append(Bullets(items=items, columns=2))
        table.decompose()

    for ul in node.find_all(["ul", "ol"]):
        items = [_tidy(_clean(li.get_text(" "))) for li in ul.find_all("li")]
        items = [i for i in items if _is_substantive(i)]
        if items:
            blocks.append(Bullets(items=items, columns=1))
        ul.decompose()

    remainder = _tidy(_dedupe_runs(_clean(node.get_text(" "))))
    remainder = _fill(remainder, filler, counters)
    for chunk in _split_sentencey(remainder):
        chunk = _tidy(chunk)
        if _is_substantive(chunk):
            blocks.append(Para(text=chunk))
    return blocks


def _fill(text: str, filler: PlaceholderFiller, counters: dict) -> str:
    """Replace LiveCareer scrub placeholders with stable invented values."""

    def _co(_m):
        idx = counters.setdefault("company", 0)
        counters["company"] = idx + 1
        return filler.company(idx)

    def _ci(_m):
        idx = counters.setdefault("city", 0)
        counters["city"] = idx + 1
        return filler.city_state(idx)

    text = _PLACEHOLDER_CITY.sub(_ci, text)
    text = _PLACEHOLDER_COMPANY.sub(_co, text)
    return text


def _parse_entry(par: Tag, filler: PlaceholderFiller, counters: dict) -> Entry:
    company = _text_of(par.find(class_="companyname"))
    jobtitle = _text_of(par.find(class_="jobtitle")) or _text_of(par.find(class_="degree"))
    city = _text_of(par.find(class_="jobcity"))
    state = _text_of(par.find(class_="jobstate"))
    # Some entries carry the connector inside the span text ("06/2004 to"),
    # which double-joins to "06/2004 to to".
    date_parts = [
        _tidy(d) for d in (_text_of(t) for t in par.find_all(class_="jobdates")) if d
    ]
    date_parts = [d for d in date_parts if _is_substantive(d)]
    dates = " to ".join(dict.fromkeys(date_parts))

    for cls in ("companyname", "jobtitle", "degree", "jobcity", "jobstate", "jobdates"):
        for t in par.find_all(class_=cls):
            t.decompose()
    # The wrapper survives its children and contributes the " to " separator.
    for t in par.find_all(class_="datesWrapper"):
        t.decompose()

    location = ", ".join(x for x in (city, state) if x)
    entry = Entry(
        title=_fill(jobtitle, filler, counters),
        org=_fill(company, filler, counters) if company else "",
        location=_fill(location, filler, counters) if location else "",
        dates=dates,
        body=_parse_blocks(par, filler, counters),
    )
    return entry


def parse_resume(html: str, resume_id: int, category: str) -> Resume:
    """Parse one LiveCareer `Resume_html` blob into a structured document."""
    soup = BeautifulSoup(html, "lxml")
    filler = PlaceholderFiller(resume_id)
    counters: dict = {}
    doc = Resume(resume_id=resume_id, category=category)

    name_div = soup.find(class_="name")
    if name_div is not None:
        raw = _clean(name_div.get_text(" "))
        # The `name` field of this corpus holds the resume's headline, not a
        # person's name (real names were stripped by LiveCareer). Keep it as
        # the headline; the pseudonym is attached later by build.py.
        doc.headline = _fill(raw, filler, counters)
        name_div.decompose()

    for sec in soup.find_all(class_="section"):
        title = _text_of(sec.find(class_="sectiontitle"))
        st = sec.find(class_="sectiontitle")
        if st is not None:
            st.decompose()
        if not title:
            continue

        section = Section(title=title)
        paragraphs = [p for p in sec.find_all(class_="paragraph")]
        if not paragraphs:
            paragraphs = [sec]

        for par in paragraphs:
            if par.find(class_="companyname") or par.find(class_="jobtitle") or par.find(
                class_="degree"
            ):
                entry = _parse_entry(par, filler, counters)
                if entry.title or entry.org or entry.body:
                    section.items.append(entry)
            else:
                section.items.extend(_parse_blocks(par, filler, counters))

        section.items = [
            i for i in section.items
            if not (isinstance(i, Para) and not i.text)
            and not (isinstance(i, Bullets) and not i.items)
        ]
        if section.items:
            doc.sections.append(section)

    return doc
