"""Corpus construction for hrsim4: sample -> pseudonymize -> render all arms.

Sampling is stratified by occupation and gated on `qc.eligible`, so a resume
that parsed badly never reaches the corpus. Pseudonyms and contact details are
derived deterministically from `resume_id`, which means every typographic arm
of a given resume carries byte-identical content and a rebuild reproduces the
same corpus exactly.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from typo_eval.hrsim4 import qc
from typo_eval.hrsim4.parse import Resume, parse_resume
from typo_eval.hrsim4.render import ResumeRenderer
from typo_eval.render import normalize_font_sizes

OCCUPATIONS = [
    "INFORMATION-TECHNOLOGY", "ENGINEERING", "FINANCE",
    "HEALTHCARE", "SALES", "TEACHER",
]

BASE_PT = 10  # probe size for ag_height normalization
REFERENCE_ARM = "T3_sans"  # the within-image comparison baseline


@dataclass(frozen=True)
class Arm:
    id: str
    font_file: str
    family: str


# Six typefaces carry the presentation question. The three Roboto weights are
# one family at three stroke weights: they vary ink density while holding
# typeface identity fixed, which is the only way to tell "OpenDyslexic is read
# as a dyslexia signal" apart from "lighter pages are judged more leniently".
ARMS: List[Arm] = [
    Arm("T1_serif", "LiberationSerif-Regular.ttf", "liberation-serif"),
    Arm("T3_sans", "LiberationSans-Regular.ttf", "liberation-sans"),
    Arm("T5_mono", "LiberationMono-Regular.ttf", "liberation-mono"),
    Arm("T7_comic", "ComicNeue-Regular.ttf", "comic-neue"),
    Arm("A1_opendyslexic", "OpenDyslexic-Regular.otf", "opendyslexic"),
    Arm("W1_roboto_light", "Roboto-Light.ttf", "roboto"),
    Arm("W2_roboto_regular", "Roboto-Regular.ttf", "roboto"),
    Arm("W3_roboto_bold", "Roboto-Bold.ttf", "roboto"),
]

# Androgynous given names with unremarkable Anglo surnames: the wave-1 corpus
# holds demographic signal constant so the font effect is isolated. Wave 2
# re-renders the calibrated boundary items with Gaddis-2017 racialized names.
_FIRST = [
    "Rowan", "Riley", "Morgan", "Jordan", "Hayden", "Taylor", "Charlie", "Peyton",
    "Casey", "Cameron", "Devon", "Finley", "Quinn", "Reese", "Sawyer", "Emerson",
    "Harper", "Avery", "Blake", "Dakota", "Ellis", "Frankie", "Greer", "Hollis",
    "Jamie", "Kendall", "Lennon", "Marlowe", "Noel", "Oakley", "Parker", "Remy",
    "Shea", "Tatum", "Wren", "Arden",
]
_LAST = [
    "Bennett", "Hale", "Avery", "Frost", "Brooks", "Vance", "Lane", "Whitman",
    "Doyle", "Marlowe", "Ellis", "Marsh", "Calloway", "Dunn", "Everett", "Fletcher",
    "Grady", "Hollis", "Ingram", "Keller", "Larkin", "Mercer", "Nolan", "Prescott",
    "Quill", "Rhodes", "Sutton", "Thatcher", "Vaughn", "Wexler", "Ashby", "Boone",
]


def _seed(resume_id: int, salt: str) -> int:
    return int(hashlib.sha256(f"hrsim4:{salt}:{resume_id}".encode()).hexdigest()[:12], 16)


def assign_identity(resume_id: int, taken: set) -> Dict[str, str]:
    """Deterministic unique pseudonym plus plausible contact details."""
    s = _seed(resume_id, "name")
    for bump in range(len(_FIRST) * len(_LAST)):
        first = _FIRST[(s + bump) % len(_FIRST)]
        last = _LAST[(s // 37 + bump * 7) % len(_LAST)]
        full = f"{first} {last}"
        if full not in taken:
            break
    taken.add(full)
    p = _seed(resume_id, "phone")
    # 555-01xx is the reserved fictional range.
    phone = f"(555) 01{p % 100:02d}-{(p // 100) % 10000:04d}"
    return {
        "name": full,
        "email": f"{first.lower()}.{last.lower()}@example.com",
        "phone": phone,
    }


def load_pool(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[df.Category.isin(OCCUPATIONS)].copy()


def build_corpus(
    csv_path: Path,
    per_occupation: int,
    fonts_dir: Path,
    out_dir: Path,
    render: bool = True,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    pool = load_pool(csv_path)
    taken: set = set()
    chosen: List[Resume] = []
    rejects: List[dict] = []

    # ag_height normalization must be resolved BEFORE selection: the admission
    # probe has to measure pages at the same scale the real render will use, or
    # it silently under-counts and admits resumes that overflow.
    paths = [fonts_dir / a.font_file for a in ARMS]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Font not found: {p}")
    normalized = normalize_font_sizes(sorted(set(paths)), BASE_PT, method="ag_height")
    scales = {str(p): normalized[str(p)] / BASE_PT for p in paths}

    # Each revision must start from an empty corpus directory. Leaving the
    # previous revision's files behind makes `audit` read a mixture of two
    # builds -- which is how a clean 240-resume corpus reported 277 resumes
    # and duplicate pseudonyms.
    for stale in ("resumes", "rendered"):
        d = out_dir / stale
        if d.exists():
            shutil.rmtree(d)

    # Admission is gated on the reference arm's ACTUAL page count, not on a
    # character-count proxy. Two resumes of the same length can occupy
    # different numbers of pages depending on how bullet-heavy they are, so a
    # blunt character cap tight enough to guarantee two pages throws away far
    # more of the pool than necessary -- and biases what survives toward short,
    # junior resumes. Laying out the reference arm costs a few milliseconds and
    # is exact.
    ref_font = fonts_dir / next(a.font_file for a in ARMS if a.id == REFERENCE_ARM)

    for occ in OCCUPATIONS:
        cand = pool[pool.Category == occ].sort_values("ID")
        kept = 0
        for _, row in cand.iterrows():
            if kept >= per_occupation:
                break
            doc = parse_resume(row.Resume_html, int(row.ID), occ)
            # Attach the identity BEFORE checking. The name and contact lines
            # are part of what gets rendered, so measuring length and page
            # count without them under-counts by a couple of lines -- enough to
            # admit resumes that then overflow onto a third page.
            ident = assign_identity(int(row.ID), taken)
            doc.name = ident["name"]
            doc.contact = [ident["email"], ident["phone"]]

            verdict = qc.eligible(doc)
            reason = ";".join(verdict.reasons) if not verdict.ok else ""
            if verdict.ok:
                probe = ResumeRenderer(ref_font, base_scale=scales[str(ref_font)])
                n_ref_pages = len(probe.paginate(probe.layout(doc)))
                if n_ref_pages > qc.MAX_REFERENCE_PAGES:
                    reason = f"reference_arm_pages={n_ref_pages}"
            if reason:
                taken.discard(ident["name"])  # release the name for the next resume
                rejects.append({"resume_id": int(row.ID), "category": occ,
                                "reasons": reason})
                continue
            chosen.append(doc)
            kept += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rejects).to_csv(out_dir / "rejected_resumes.csv", index=False)

    records: List[dict] = []
    docs = chosen[:limit] if limit else chosen
    items_dir = out_dir / "resumes"
    items_dir.mkdir(parents=True, exist_ok=True)

    for doc in tqdm(docs, desc="hrsim4 corpus"):
        text = doc.plain_text()
        (items_dir / f"{doc.resume_id}.json").write_text(
            json.dumps({"resume_id": doc.resume_id, "category": doc.category,
                        "name": doc.name, "plain_text": text}, indent=1)
        )
        for arm in ARMS:
            fp = fonts_dir / arm.font_file
            renderer = ResumeRenderer(fp, base_scale=scales[str(fp)])
            paths_out: List[str] = []
            if render:
                pages = renderer.render(doc)
                d = out_dir / "rendered" / str(doc.resume_id)
                d.mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(pages, 1):
                    p = d / f"{arm.id}_p{i}.png"
                    img.save(p)
                    paths_out.append(str(p))
                n_pages = len(pages)
            else:
                # Page count comes from layout + pagination alone; drawing the
                # pixels is the expensive part and it cannot change how many
                # pages the content occupies. This lets the admission-threshold
                # search converge over the full corpus in seconds rather than
                # re-rendering thousands of images per revision.
                n_pages = len(renderer.paginate(renderer.layout(doc)))
            records.append({
                "resume_id": doc.resume_id, "category": doc.category,
                "arm": arm.id, "family": arm.family,
                "pages": n_pages, "text_chars": len(text),
                "font_scale": round(scales[str(fp)], 4),
                "image_paths": "|".join(paths_out),
            })

    manifest = pd.DataFrame(records)
    manifest.to_csv(out_dir / "render_manifest.csv", index=False)
    return manifest
