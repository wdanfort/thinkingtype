"""Gate-drift comparison: did the gate's behavior change between two runs?

Compares two gates runs (different model versions, prompt variants, or
vendors) on the items they share. Because both runs carry repeated-sample
text calibrations (and optionally vision arms), the comparison is grounded
at the decision level:

- decision flips: items whose majority text-mode decision differs A -> B,
  with direction coded by favorability (positive = B more favorable to the
  person judged);
- boundary migration: items entering/leaving the calibration band;
- modality drift (when both runs have vision data): change in the paired
  image-minus-text delta per shared variant.

Typical uses: regression-test a model upgrade (same config, new model_text /
model_vision), quantify a prompt change (v1 vs rubric), or compare vendors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from typo_eval.gates.engine import load_records
from typo_eval.gates.prompts import get_gate_spec

logger = logging.getLogger(__name__)


def _load_calibration(run_dir: Path, provider: Optional[str]) -> pd.DataFrame:
    frames = []
    for jsonl in sorted((run_dir / "calibration").glob("*.jsonl")):
        if provider and jsonl.stem != provider:
            continue
        frames.append(load_records(jsonl))
    if not frames:
        raise FileNotFoundError(f"No calibration JSONL under {run_dir}/calibration")
    df = pd.concat(frames, ignore_index=True)
    ok = df[df["parsed_response"].notna()]
    out = (
        ok.groupby(["provider", "model", "gate", "item_id"])
        .agg(n=("parsed_response", "size"), n_yes=("parsed_response", "sum"))
        .reset_index()
    )
    out["p_yes"] = out["n_yes"] / out["n"]
    return out


def _load_vision_delta(run_dir: Path, provider: Optional[str]) -> pd.DataFrame:
    """Per item x variant image p_yes, for runs that have a vision arm."""
    frames = []
    vision_dir = run_dir / "vision"
    if vision_dir.exists():
        for jsonl in sorted(vision_dir.glob("*.jsonl")):
            if provider and jsonl.stem != provider:
                continue
            frames.append(load_records(jsonl))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    ok = df[df["parsed_response"].notna()]
    if ok.empty:
        return pd.DataFrame()
    out = (
        ok.groupby(["provider", "gate", "item_id", "variant_id"])
        .agg(n=("parsed_response", "size"), n_yes=("parsed_response", "sum"))
        .reset_index()
    )
    out["p_image"] = out["n_yes"] / out["n"]
    return out[["provider", "gate", "item_id", "variant_id", "p_image"]]


def _fav_sign(gate: str) -> int:
    try:
        return 1 if get_gate_spec(gate).yes_is_favorable else -1
    except ValueError:
        return 1  # unknown gate in old logs: report raw direction


def compare_runs(
    run_a: Path,
    run_b: Path,
    out_dir: Path,
    label_a: str,
    label_b: str,
    provider: Optional[str] = None,
    band: tuple = (0.15, 0.85),
) -> Path:
    """Write a drift report comparing run B against run A."""
    out_dir.mkdir(parents=True, exist_ok=True)

    a = _load_calibration(run_a, provider).rename(
        columns={"p_yes": "p_a", "model": "model_a", "n": "n_a"}
    )
    b = _load_calibration(run_b, provider).rename(
        columns={"p_yes": "p_b", "model": "model_b", "n": "n_b"}
    )
    merged = a.merge(
        b, on=["provider", "gate", "item_id"], how="inner", suffixes=("_a", "_b")
    )
    if merged.empty:
        raise ValueError(
            "Runs share no (provider, gate, item) triples; nothing to compare"
        )

    merged["fav_sign"] = merged["gate"].map(_fav_sign)
    merged["maj_a"] = (merged["p_a"] > 0.5).astype(int)
    merged["maj_b"] = (merged["p_b"] > 0.5).astype(int)
    merged["flip"] = (merged["maj_a"] != merged["maj_b"]).astype(int)
    merged["flip_dir_fav"] = (merged["maj_b"] - merged["maj_a"]) * merged["fav_sign"]
    merged["delta_p_fav"] = (merged["p_b"] - merged["p_a"]) * merged["fav_sign"]
    lo, hi = band
    merged["in_band_a"] = (merged["p_a"] >= lo) & (merged["p_a"] <= hi)
    merged["in_band_b"] = (merged["p_b"] >= lo) & (merged["p_b"] <= hi)
    merged.to_csv(out_dir / "drift_items.csv", index=False)

    summary = (
        merged.groupby(["provider", "gate"])
        .agg(
            n_shared=("item_id", "size"),
            model_a=("model_a", "first"),
            model_b=("model_b", "first"),
            flips=("flip", "sum"),
            net_fav=("flip_dir_fav", "sum"),
            mean_delta_p_fav=("delta_p_fav", "mean"),
            entered_band=("in_band_b", lambda s: int((s & ~merged.loc[s.index, "in_band_a"]).sum())),
            left_band=("in_band_a", lambda s: int((s & ~merged.loc[s.index, "in_band_b"]).sum())),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "drift_summary.csv", index=False)

    # Modality drift where both runs have vision arms
    va = _load_vision_delta(run_a, provider)
    vb = _load_vision_delta(run_b, provider)
    modality_lines: List[str] = []
    if not va.empty and not vb.empty:
        va = va.merge(
            a[["provider", "gate", "item_id", "p_a"]], on=["provider", "gate", "item_id"]
        )
        vb = vb.merge(
            b[["provider", "gate", "item_id", "p_b"]], on=["provider", "gate", "item_id"]
        )
        va["mod_a"] = va["p_image"] - va["p_a"]
        vb["mod_b"] = vb["p_image"] - vb["p_b"]
        mod = va.merge(
            vb,
            on=["provider", "gate", "item_id", "variant_id"],
            how="inner",
        )
        if not mod.empty:
            mod["fav_sign"] = mod["gate"].map(_fav_sign)
            mod["mod_a_fav"] = mod["mod_a"] * mod["fav_sign"]
            mod["mod_b_fav"] = mod["mod_b"] * mod["fav_sign"]
            mod_summary = (
                mod.groupby(["provider", "gate", "variant_id"])
                .agg(
                    n=("item_id", "size"),
                    modality_fav_a=("mod_a_fav", "mean"),
                    modality_fav_b=("mod_b_fav", "mean"),
                )
                .reset_index()
            )
            mod_summary["change"] = (
                mod_summary["modality_fav_b"] - mod_summary["modality_fav_a"]
            )
            mod_summary.to_csv(out_dir / "drift_modality.csv", index=False)
            modality_lines = [
                "\n## Modality drift (image-minus-text delta, favorability-signed)\n",
                mod_summary.to_markdown(index=False),
            ]

    lines = [
        f"# Gate drift: {label_b} vs {label_a}\n",
        f"Baseline (A): `{label_a}` — comparison (B): `{label_b}`.",
        "Positive net/mean values mean B decides more favorably for the "
        "person judged than A on the same items.\n",
        "## Text-mode decision drift\n",
        summary.to_markdown(index=False),
    ]
    flips = merged[merged["flip"] == 1]
    if not flips.empty:
        lines.append("\n## Flipped items\n")
        lines.append(
            flips[
                ["provider", "gate", "item_id", "p_a", "p_b", "flip_dir_fav"]
            ].to_markdown(index=False)
        )
    lines.extend(modality_lines)
    lines.append("")
    report = out_dir / "drift_report.md"
    report.write_text("\n".join(lines))
    logger.info(f"Drift report written to {report}")
    return report
