"""Analysis for hrsim4.

Three axes, where v1-v3 had two:

  1. MODALITY   image vs the text baseline, per arm
  2. TYPEFACE   each typeface vs the sans reference, modality held constant
  3. INK        the Roboto Light/Regular/Bold ladder -- same typeface, three
                stroke weights. This is the axis v1-v3 lacked, and it is what
                lets us answer the standing objection to the whole programme:
                is the model reacting to the typeface, or just to how much dark
                ink is on the page? The ladder brackets all five typefaces on
                measured ink coverage, so its slope predicts how much of each
                typeface's effect is explained by ink alone.

All CIs are resume-clustered percentile bootstraps (a resume contributes 9
correlated observations, so row-level resampling would understate the spread).
Refusals and parse failures are dropped from rates and reported separately.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from typo_eval.hrsim.analyze import _bootstrap_clustered_mean
from typo_eval.hrsim4.config import INK_LADDER_ARMS, TEXT_ARM, TYPEFACE_ARMS

logger = logging.getLogger(__name__)

SANS_REF = "T3_sans"

# Words that would indicate the model is citing presentation as a reason. v1-v3
# found ZERO genuine hits in ~11k traces (only false positives like the employer
# "La Fontaine Bleu" matching "font"), which is the absorbed-prior signature:
# decisions move without the model ever naming the cause.
_TYPO_RE = re.compile(
    r"\b(font|typeface|typograph\w*|comic sans|opendyslexic|open dyslexic|"
    r"serif|monospac\w*|italic|bold(?:face)?|legib\w*|readab\w*|formatt?\w*|"
    r"layout|presentation|unprofessional[- ]looking)\b",
    re.IGNORECASE,
)


def load_all_legs(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(run_dir).glob("raw_*.jsonl")):
        rows = []
        for line in path.open():
            try:
                rows.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise FileNotFoundError(f"no raw_*.jsonl under {run_dir}")
    return pd.concat(frames, ignore_index=True)


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    """Usable decisions only. Deduplicates on the record key, keeping the last
    write, so a resumed run that re-ran a cell doesn't double-count it."""
    d = df[df.parsed_decision.notna()].copy()
    d["parsed_decision"] = d["parsed_decision"].astype(int)
    return d.drop_duplicates(
        subset=["leg", "item_id", "arm", "variant_id", "repeat_idx"], keep="last"
    )


def completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Refusals, parse failures and transport errors per leg."""
    rows = []
    for leg, g in df.groupby("leg"):
        err = g.parse_error.fillna("")
        rows.append({
            "leg": leg, "records": len(g),
            "ok": int(g.parsed_decision.notna().sum()),
            "refusals": int((err == "refusal").sum()),
            "no_match": int((err == "no_match").sum()),
            "exceptions": int(err.str.startswith("exception").sum()),
        })
    return pd.DataFrame(rows)


def accept_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Accept rate per leg x arm, with resume-clustered CI."""
    rows = []
    for (leg, variant), g in _ok(df).groupby(["leg", "variant_id"]):
        lo, hi = _bootstrap_clustered_mean(g, "parsed_decision")
        rows.append({
            "leg": leg, "variant_id": variant, "n": len(g),
            "accept_rate": float(g.parsed_decision.mean()),
            "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows).sort_values(["leg", "variant_id"])


def _paired(df: pd.DataFrame, base_variant: str, subset: Optional[str] = None
            ) -> Dict[str, pd.DataFrame]:
    """Item-level paired frames: each variant joined to `base_variant` on the
    same item, so every comparison is within-resume."""
    d = _ok(df)
    if subset == "sensitive":
        d = d[d.boundary_sensitive]
    elif subset == "locked":
        d = d[~d.boundary_sensitive]
    out = {}
    for leg, g in d.groupby("leg"):
        base = (g[g.variant_id == base_variant]
                .groupby("item_id").parsed_decision.mean().rename("base"))
        for variant, gv in g.groupby("variant_id"):
            if variant == base_variant:
                continue
            v = gv.groupby("item_id").parsed_decision.mean().rename("variant")
            m = pd.concat([base, v], axis=1).dropna()
            if m.empty:
                continue
            m["delta"] = m["variant"] - m["base"]
            m["resume_id"] = [int(str(i).split("_")[0]) for i in m.index]
            out[(leg, variant)] = m
    return out


def _delta_table(paired: Dict, label: str, n_boot: int = 2000, seed: int = 42
                 ) -> pd.DataFrame:
    rows = []
    for (leg, variant), m in paired.items():
        lo, hi = _bootstrap_clustered_mean(m, "delta", n_boot=n_boot, seed=seed)
        rows.append({
            "leg": leg, "comparison": label, "variant_id": variant,
            "n_items": len(m),
            "base_rate": float(m["base"].mean()),
            "variant_rate": float(m["variant"].mean()),
            "delta_pp": 100 * float(m["delta"].mean()),
            "ci_lo_pp": 100 * lo, "ci_hi_pp": 100 * hi,
            "significant": bool(lo > 0 or hi < 0),
        })
    cols = ["leg", "comparison", "variant_id", "n_items", "base_rate",
            "variant_rate", "delta_pp", "ci_lo_pp", "ci_hi_pp", "significant"]
    if not rows:  # e.g. a subset with no items in it
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(["leg", "delta_pp"])


def modality_effects(df: pd.DataFrame, subset: Optional[str] = None) -> pd.DataFrame:
    """Axis 1: each rendered arm vs the text baseline."""
    return _delta_table(_paired(df, TEXT_ARM, subset), "image_vs_text")


def typeface_effects(df: pd.DataFrame, subset: Optional[str] = None) -> pd.DataFrame:
    """Axis 2: each rendered arm vs the sans reference, modality held constant."""
    t = _delta_table(_paired(df, SANS_REF, subset), "vs_sans")
    return t[t.variant_id != TEXT_ARM] if len(t) else t


def ink_slope(df: pd.DataFrame, ink_by_arm: Dict[str, float],
              n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Axis 3: how many pp of accept rate does one unit of ink coverage buy?

    Fits accept ~ ink across the three ladder rungs ONLY (same typeface, so the
    only thing varying is stroke weight), then reports, for each typeface, how
    much of its measured vs-sans effect the ink slope alone would predict. A
    typeface whose effect is far from its ink prediction is doing something the
    ink account cannot explain.
    """
    d = _ok(df)
    rows = []
    for leg, g in d.groupby("leg"):
        ladder = g[g.variant_id.isin(INK_LADDER_ARMS)]
        if ladder.variant_id.nunique() < 2:
            continue
        # Resume-clustered bootstrap of the OLS slope over the ladder rungs.
        per_item = (ladder.groupby(["item_id", "variant_id"]).parsed_decision.mean()
                    .reset_index())
        per_item["ink"] = per_item.variant_id.map(ink_by_arm)
        per_item["resume_id"] = [int(str(i).split("_")[0]) for i in per_item.item_id]
        groups = {c: gg for c, gg in per_item.groupby("resume_id")}
        clusters = list(groups)
        rng = np.random.default_rng(seed)

        def _slope(frame):
            x, y = frame["ink"].to_numpy(), frame["parsed_decision"].to_numpy()
            if len(np.unique(x)) < 2:
                return np.nan
            return float(np.polyfit(x, y, 1)[0])

        point = _slope(per_item)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            samp = rng.choice(clusters, size=len(clusters), replace=True)
            boots[b] = _slope(pd.concat([groups[c] for c in samp], ignore_index=True))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])

        ink_ref = ink_by_arm.get(SANS_REF, np.nan)
        tf = typeface_effects(df)
        tf = tf[(tf.leg == leg) & (tf.variant_id.isin(TYPEFACE_ARMS))]
        for _, r in tf.iterrows():
            d_ink = ink_by_arm.get(r.variant_id, np.nan) - ink_ref
            predicted = 100 * point * d_ink
            rows.append({
                "leg": leg, "variant_id": r.variant_id,
                "ink_coverage": ink_by_arm.get(r.variant_id),
                "delta_ink_vs_sans": d_ink,
                "observed_pp": r.delta_pp,
                "ink_predicted_pp": predicted,
                "unexplained_pp": r.delta_pp - predicted,
                "ink_slope_pp_per_unit": 100 * point,
                "slope_ci_lo": 100 * lo, "slope_ci_hi": 100 * hi,
            })
    return pd.DataFrame(rows)


def sensitivity_check(df: pd.DataFrame) -> pd.DataFrame:
    """Does gate 3's JD-perturbation sensitivity actually predict font
    sensitivity? If it does, later runs can restrict to the sensitive subset and
    halve their cost; if not, the proxy is not usable for selection."""
    rows = []
    d = _ok(df)
    for leg, g in d.groupby("leg"):
        for label, sub in (("sensitive", g[g.boundary_sensitive]),
                           ("locked", g[~g.boundary_sensitive])):
            piv = sub.pivot_table(index="item_id", columns="variant_id",
                                  values="parsed_decision", aggfunc="mean")
            if piv.empty:
                continue
            image_cols = [c for c in piv.columns if c != TEXT_ARM]
            moved = (piv[image_cols].nunique(axis=1) > 1)
            rows.append({
                "leg": leg, "group": label, "n_items": len(piv),
                "items_where_arms_disagree": int(moved.sum()),
                "share": float(moved.mean()),
            })
    return pd.DataFrame(rows)


def typography_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """Absorbed-prior check: does the model ever NAME presentation as a reason?"""
    d = df[df.raw_response.notna()]
    rows = []
    for leg, g in d.groupby("leg"):
        hits = g[g.raw_response.str.contains(_TYPO_RE, na=False)]
        rows.append({"leg": leg, "traces": len(g), "keyword_hits": len(hits),
                     "hit_rate": len(hits) / max(1, len(g))})
    return pd.DataFrame(rows)


def mention_examples(df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    """Keyword hits with context, for manual true/false-positive triage.
    v1-v3's hits were ALL false positives; this is what makes that checkable."""
    d = df[df.raw_response.notna()]
    out = []
    for _, r in d.iterrows():
        for m in _TYPO_RE.finditer(str(r.raw_response)):
            s, e = max(0, m.start() - 70), min(len(r.raw_response), m.end() + 70)
            out.append({"leg": r.leg, "variant_id": r.variant_id,
                        "item_id": r.item_id, "match": m.group(0),
                        "context": "..." + r.raw_response[s:e].replace("\n", " ") + "..."})
            if len(out) >= limit:
                return pd.DataFrame(out)
    return pd.DataFrame(out)


def load_ink_by_arm(qc_report: Path) -> Dict[str, float]:
    """Measured ink coverage per arm, parsed from loop 1's QC report."""
    text = Path(qc_report).read_text()
    ink: Dict[str, float] = {}
    for line in text.splitlines():
        if "ink_ladder_monotonic" in line or "ink_ladder_brackets_typefaces" in line:
            for arm, val in re.findall(r"'([A-Za-z0-9_]+)':\s*([0-9.]+)", line):
                ink[arm] = float(val)
    return ink


def analyze_run(run_dir: Path, corpus_dir: Path, out_dir: Optional[Path] = None
                ) -> Dict[str, pd.DataFrame]:
    run_dir, corpus_dir = Path(run_dir), Path(corpus_dir)
    out_dir = Path(out_dir or run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_all_legs(run_dir)
    ink = load_ink_by_arm(corpus_dir / "qc_report.txt")

    tables = {
        "completeness": completeness(raw),
        "accept_rates": accept_rates(raw),
        "modality_effects": modality_effects(raw),
        "modality_effects_sensitive": modality_effects(raw, subset="sensitive"),
        "typeface_effects": typeface_effects(raw),
        "typeface_effects_sensitive": typeface_effects(raw, subset="sensitive"),
        "ink_slope": ink_slope(raw, ink),
        "sensitivity_check": sensitivity_check(raw),
        "typography_mentions": typography_mentions(raw),
        "mention_examples": mention_examples(raw),
    }
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    logger.info(f"wrote {len(tables)} tables to {out_dir}")
    return tables
