"""Quality gates for the hrsim4 corpus.

Every criterion here traces to a specific defect found in the 2026-08-04 audit
of hrsim_v1/v2/v3. The corpus is rebuilt and re-checked until all of them pass;
`audit()` is the stopping condition, not a diagnostic.

Per-resume gates (`eligible`) run during sampling, so a defective resume is
never admitted. Corpus-level gates (`audit`) run over the finished render
manifest and check the properties that only exist across arms -- content
parity, no silent downscaling, a working ink ladder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from typo_eval.hrsim4.parse import Bullets, Entry, Para, Resume

# --- thresholds ------------------------------------------------------------
MIN_SECTIONS = 3
MIN_ENTRIES = 2
MIN_BULLETS = 5
MIN_CHARS = 1800
# Length is bounded by PAGES, not characters -- `build_corpus` lays out the
# reference arm and rejects anything over MAX_REFERENCE_PAGES, which is exact
# where a character count is only a proxy. A character cap tight enough to
# guarantee two pages was rejecting resumes that actually fit (it cost 64 of
# Healthcare's candidates, whose clinical resumes are long but lay out
# compactly). MAX_CHARS survives only to exclude absurd outliers.
MAX_CHARS = 9000
MAX_IMAGE_DIM = 1568          # Anthropic downscales above this
MAX_REFERENCE_PAGES = 2       # a real resume is 1-2 pages; enforced at admission
MAX_PAGES = 4                 # OD needs ~1.5x the space of sans for the same text

_EXPERIENCE_RE = re.compile(r"experience|employment|work history|professional background", re.I)
_CREDENTIAL_RE = re.compile(r"education|training|certification|licens|qualification|credential", re.I)

# Defects that must not survive into the corpus.
_DEFECTS: Dict[str, re.Pattern] = {
    "scrub_placeholder": re.compile(r"\bCompany Name\b|\bCity\s*,\s*State\b"),
    "template_placeholder": re.compile(r"\[(?:Job Title|Company Name|Timeframe|School Name|Degree)\]"),
    "date_join_bug": re.compile(r"\bto\s+to\b"),
    "doubled_punctuation": re.compile(r",\s*,|\s+,\s*$"),
    "mojibake": re.compile(r"Ã.|â€|�"),
    "leading_separator": re.compile(r"^\s*[,;:]", re.M),
}

# A line that trails off in a connector is the signature of source truncation.
_DANGLING_RE = re.compile(r"\b(?:to|and|or|of|for|with|but|the|a|an|in|on|at|from|by)\s*$", re.I)


@dataclass
class Verdict:
    ok: bool
    reasons: List[str] = field(default_factory=list)


def _counts(doc: Resume) -> Dict[str, int]:
    entries = bullets = 0
    for sec in doc.sections:
        for item in sec.items:
            if isinstance(item, Entry):
                entries += 1
                bullets += sum(len(b.items) for b in item.body if isinstance(b, Bullets))
            elif isinstance(item, Bullets):
                bullets += len(item.items)
    return {"entries": entries, "bullets": bullets, "sections": len(doc.sections)}


def _all_text_units(doc: Resume) -> List[str]:
    out: List[str] = []
    for sec in doc.sections:
        for item in sec.items:
            blocks = item.body if isinstance(item, Entry) else [item]
            for b in blocks:
                if isinstance(b, Para):
                    out.append(b.text)
                elif isinstance(b, Bullets):
                    out.extend(b.items)
    return out


def eligible(doc: Resume) -> Verdict:
    """Per-resume admission gate."""
    reasons: List[str] = []
    text = doc.plain_text()
    c = _counts(doc)

    if c["sections"] < MIN_SECTIONS:
        reasons.append(f"sections<{MIN_SECTIONS}({c['sections']})")
    if c["entries"] < MIN_ENTRIES:
        reasons.append(f"entries<{MIN_ENTRIES}({c['entries']})")
    if c["bullets"] < MIN_BULLETS:
        reasons.append(f"bullets<{MIN_BULLETS}({c['bullets']})")
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        reasons.append(f"chars_out_of_band({len(text)})")

    titles = " ".join(s.title for s in doc.sections)
    if not _EXPERIENCE_RE.search(titles):
        reasons.append("no_experience_section")
    # v3 showed the model no credential at all for 41% of resumes because
    # truncation cut Education. Require it to be present and parsed.
    if not _CREDENTIAL_RE.search(titles):
        reasons.append("no_credential_section")

    for name, pat in _DEFECTS.items():
        if pat.search(text):
            reasons.append(name)

    units = _all_text_units(doc)
    if units and sum(1 for u in units if _DANGLING_RE.search(u)) > 0:
        reasons.append("dangling_connector")

    return Verdict(ok=not reasons, reasons=reasons)


# --------------------------------------------------------------- corpus gates


@dataclass
class Gate:
    name: str
    ok: bool
    detail: str


def font_charset(font_path: Path) -> set:
    """Every codepoint the font can actually draw."""
    from fontTools.ttLib import TTFont

    tt = TTFont(str(font_path), fontNumber=0, lazy=True)
    covered: set = set()
    for table in tt["cmap"].tables:
        covered.update(table.cmap.keys())
    tt.close()
    return covered


def glyph_coverage(texts: List[str], font_paths: Dict[str, Path]) -> Dict[str, set]:
    """Characters used by the corpus that a given arm's font cannot render.

    A missing glyph does not fail loudly -- it draws a tofu box. That makes the
    same resume literally different content in different arms, which is exactly
    the confound the whole design is trying to avoid, so this is a hard gate.
    """
    used = set()
    for t in texts:
        used.update(ord(c) for c in t)
    # Characters the renderer itself draws, which never appear in plain_text:
    # the bullet marker and the em dash joining job title to employer.
    used.update(ord(c) for c in "•—")
    used.discard(ord("\n"))
    missing: Dict[str, set] = {}
    for arm, path in font_paths.items():
        gap = used - font_charset(path)
        if gap:
            missing[arm] = gap
    return missing


def audit(
    manifest: pd.DataFrame,
    corpus_dir: Path,
    expected_arms: int,
    per_occupation: int,
    measure_images: bool = True,
    font_paths: Optional[Dict[str, Path]] = None,
) -> List[Gate]:
    """Corpus-level gates. All must pass before the corpus is usable."""
    gates: List[Gate] = []

    def add(name: str, ok: bool, detail: str) -> None:
        gates.append(Gate(name, bool(ok), detail))

    n_res = manifest.resume_id.nunique()
    add("resume_count",
        n_res == per_occupation * manifest.category.nunique(),
        f"{n_res} resumes across {manifest.category.nunique()} occupations")

    per_occ = manifest.groupby("category").resume_id.nunique()
    add("occupation_balance", bool((per_occ == per_occupation).all()),
        per_occ.to_dict())

    arms_per = manifest.groupby("resume_id").arm.nunique()
    add("all_arms_rendered", bool((arms_per == expected_arms).all()),
        f"min={arms_per.min()} max={arms_per.max()} expected={expected_arms}")

    # Content parity: the same resume must carry identical text in every arm.
    spread = manifest.groupby("resume_id").text_chars.nunique()
    add("content_parity_across_arms", bool((spread == 1).all()),
        f"{int((spread > 1).sum())} resumes differ in text across arms")

    add("no_truncation", bool(manifest.text_chars.min() >= MIN_CHARS),
        f"min_chars={manifest.text_chars.min()} max_chars={manifest.text_chars.max()}")

    add("page_count_sane", bool(manifest.pages.between(1, MAX_PAGES).all()),
        manifest.groupby("arm").pages.mean().round(2).to_dict())

    ref = manifest[manifest.arm == "T3_sans"]
    add("reference_arm_is_1_2_pages", bool(ref.pages.between(1, 2).all()),
        f"T3_sans pages: {ref.pages.value_counts().sort_index().to_dict()}")

    import json
    docs = [json.loads(p.read_text()) for p in (corpus_dir / "resumes").glob("*.json")]
    names = [d["name"] for d in docs]
    add("unique_pseudonyms", len(names) == len(set(names)),
        f"{len(set(names))} unique of {len(names)}")

    texts = [d["plain_text"] for d in docs]
    add("no_residual_defects",
        not any(pat.search(t) for t in texts for pat in _DEFECTS.values()),
        f"checked {len(_DEFECTS)} patterns over {len(texts)} resumes")

    if font_paths:
        missing = glyph_coverage(texts, font_paths)
        add("glyph_coverage_all_arms", not missing,
            {a: sorted(f"U+{c:04X}" for c in cs)[:8] for a, cs in missing.items()}
            or f"all corpus characters render in all {len(font_paths)} arms")

    if measure_images:
        rows = []
        for _, r in manifest.iterrows():
            for path in str(r.image_paths).split("|"):
                if not path:
                    continue
                with Image.open(path) as im:
                    a = np.asarray(im.convert("L"))
                    rows.append({"arm": r.arm, "family": r.family, "w": im.width,
                                 "h": im.height, "ink": float((a < 128).mean())})
        img = pd.DataFrame(rows)
        add("no_downscaling",
            bool((img.w <= MAX_IMAGE_DIM).all() and (img.h <= MAX_IMAGE_DIM).all()),
            f"max_w={img.w.max()} max_h={img.h.max()} threshold={MAX_IMAGE_DIM}")
        add("uniform_page_geometry", img.w.nunique() == 1 and img.h.nunique() == 1,
            f"w={sorted(img.w.unique())} h={sorted(img.h.unique())}")

        # The ink ladder must actually be a ladder, or it cannot identify the
        # ink slope that the whole weight-ladder design exists to estimate.
        ladder = img[img.family == "roboto"].groupby("arm").ink.mean()
        order = ["W1_roboto_light", "W2_roboto_regular", "W3_roboto_bold"]
        if all(o in ladder.index for o in order):
            vals = [ladder[o] for o in order]
            add("ink_ladder_monotonic", vals[0] < vals[1] < vals[2],
                {k: round(float(ladder[k]), 4) for k in order})
        else:
            add("ink_ladder_monotonic", False, "ladder arms missing")

        add("ink_ladder_brackets_typefaces",
            bool(ladder.min() <= img[img.family != "roboto"].groupby("arm").ink.mean().min()
                 and ladder.max() >= img[img.family != "roboto"].groupby("arm").ink.mean().max()),
            {"ladder": [round(float(ladder.min()), 4), round(float(ladder.max()), 4)],
             "typefaces": img[img.family != "roboto"].groupby("arm").ink.mean().round(4).to_dict()})

    return gates


def report(gates: List[Gate]) -> str:
    lines = []
    for g in gates:
        lines.append(f"[{'PASS' if g.ok else 'FAIL'}] {g.name}: {g.detail}")
    n_fail = sum(1 for g in gates if not g.ok)
    lines.append(f"\n{len(gates) - n_fail}/{len(gates)} gates pass"
                 + ("" if n_fail == 0 else f" -- {n_fail} FAILING"))
    return "\n".join(lines)
