"""Analysis for the gates experiment.

Everything is paired at the item level: each selected item has a text-mode
p(yes) (from calibration repeats) and an image-mode p(yes) per variant (from
vision repeats). The quantities reported:

- delta_p = p_image - p_yes(text), averaged over items, bootstrap CI.
- delta_fav = delta_p signed so positive always means "more favorable to the
  person judged" (resume/appeal: yes favorable; moderation: yes = remove =
  unfavorable). This makes "harsher" comparable across gates.
- majority flip rate: fraction of items whose majority-vote decision differs
  between text and image.
- OpenDyslexic fairness contrast: within-image A1_opendyslexic vs the sans
  reference variant, isolating the accessibility font from the modality
  switch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from typo_eval.gates.config import GatesConfig
from typo_eval.gates.engine import load_records
from typo_eval.gates.prompts import get_gate_spec

logger = logging.getLogger(__name__)

REFERENCE_VARIANT = "T3_sans"
ACCESSIBILITY_VARIANT = "A1_opendyslexic"


def _bootstrap_ci(
    values: np.ndarray, n_boot: int = 4000, alpha: float = 0.05, seed: int = 42
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean over items."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def _sign_test(deltas: np.ndarray) -> float:
    """Two-sided sign test on nonzero per-item deltas."""
    nonzero = deltas[deltas != 0]
    if len(nonzero) == 0:
        return 1.0
    n_pos = int((nonzero > 0).sum())
    return float(
        scipy_stats.binomtest(n_pos, len(nonzero), 0.5, alternative="two-sided").pvalue
    )


def _p_yes_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    ok = df[df["parsed_response"].notna()]
    out = (
        ok.groupby(group_cols)
        .agg(n=("parsed_response", "size"), n_yes=("parsed_response", "sum"))
        .reset_index()
    )
    out["p_yes"] = out["n_yes"] / out["n"]
    return out


def _fav_sign(gate: str) -> int:
    return 1 if get_gate_spec(gate).yes_is_favorable else -1


def build_paired_table(
    calibration_df: pd.DataFrame,
    vision_df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per provider x gate x item x variant with paired p_yes."""
    selected_keys = selected_df[["provider", "gate", "item_id"]].drop_duplicates()

    text = _p_yes_table(
        calibration_df, ["provider", "gate", "item_id", "scenario", "level"]
    ).rename(columns={"p_yes": "p_text", "n": "n_text"})
    text = text.merge(selected_keys, on=["provider", "gate", "item_id"])

    img = _p_yes_table(
        vision_df, ["provider", "gate", "item_id", "variant_id"]
    ).rename(columns={"p_yes": "p_image", "n": "n_image"})

    paired = img.merge(
        text[["provider", "gate", "item_id", "scenario", "level", "p_text", "n_text"]],
        on=["provider", "gate", "item_id"],
        how="inner",
    )
    paired["delta_p"] = paired["p_image"] - paired["p_text"]
    paired["fav_sign"] = paired["gate"].map(_fav_sign)
    paired["delta_fav"] = paired["delta_p"] * paired["fav_sign"]
    paired["maj_text"] = (paired["p_text"] > 0.5).astype(int)
    paired["maj_image"] = (paired["p_image"] > 0.5).astype(int)
    paired["maj_flip"] = (paired["maj_text"] != paired["maj_image"]).astype(int)
    paired["maj_flip_dir_fav"] = (
        (paired["maj_image"] - paired["maj_text"]) * paired["fav_sign"]
    )
    return paired


def _effect_rows(paired: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for key, grp in paired.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        deltas = grp["delta_p"].to_numpy()
        favs = grp["delta_fav"].to_numpy()
        lo, hi = _bootstrap_ci(favs)
        row = dict(zip(group_cols, key))
        row.update(
            n_items=len(grp),
            mean_delta_p=float(deltas.mean()),
            mean_delta_fav=float(favs.mean()),
            ci_lo_fav=lo,
            ci_hi_fav=hi,
            sign_test_p=_sign_test(favs),
            maj_flip_rate=float(grp["maj_flip"].mean()),
            maj_net_fav=int(grp["maj_flip_dir_fav"].sum()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def opendyslexic_contrast(vision_df: pd.DataFrame) -> pd.DataFrame:
    """Within-image contrast: accessibility font vs sans reference."""
    img = _p_yes_table(vision_df, ["provider", "gate", "item_id", "variant_id"])
    ref = img[img["variant_id"] == REFERENCE_VARIANT][
        ["provider", "gate", "item_id", "p_yes"]
    ].rename(columns={"p_yes": "p_ref"})
    acc = img[img["variant_id"] == ACCESSIBILITY_VARIANT][
        ["provider", "gate", "item_id", "p_yes"]
    ].rename(columns={"p_yes": "p_acc"})
    both = ref.merge(acc, on=["provider", "gate", "item_id"])
    if both.empty:
        return pd.DataFrame(), both
    both["delta_p"] = both["p_acc"] - both["p_ref"]
    both["fav_sign"] = both["gate"].map(_fav_sign)
    both["delta_fav"] = both["delta_p"] * both["fav_sign"]

    rows = []
    for cols in (["provider", "gate"], ["provider"]):
        for key, grp in both.groupby(cols):
            key = key if isinstance(key, tuple) else (key,)
            favs = grp["delta_fav"].to_numpy()
            lo, hi = _bootstrap_ci(favs)
            row = dict(zip(cols, key))
            if "gate" not in row:
                row["gate"] = "__all__"
            row.update(
                n_items=len(grp),
                mean_delta_fav=float(favs.mean()),
                ci_lo_fav=lo,
                ci_hi_fav=hi,
                sign_test_p=_sign_test(favs),
            )
            rows.append(row)
    return pd.DataFrame(rows), both


def _plot_forest(effects: pd.DataFrame, out_path: Path, title: str) -> None:
    """Forest plot of mean_delta_fav with CIs, variant x provider."""
    providers = sorted(effects["provider"].unique())
    variants = sorted(effects["variant_id"].unique())
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(variants) * len(providers))))
    colors = plt.get_cmap("tab10")
    ypos = 0
    yticks, ylabels = [], []
    for variant in variants:
        for i, provider in enumerate(providers):
            sub = effects[
                (effects["variant_id"] == variant) & (effects["provider"] == provider)
            ]
            if sub.empty:
                continue
            r = sub.iloc[0]
            ax.errorbar(
                r["mean_delta_fav"],
                ypos,
                xerr=[
                    [r["mean_delta_fav"] - r["ci_lo_fav"]],
                    [r["ci_hi_fav"] - r["mean_delta_fav"]],
                ],
                fmt="o",
                color=colors(i),
                capsize=3,
                label=provider if variant == variants[0] else None,
            )
            yticks.append(ypos)
            ylabels.append(f"{variant} / {provider}")
            ypos += 1
        ypos += 0.6
    ax.axvline(0, color="gray", lw=1, ls="--")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Δ favorability (image − text), + favors the person judged")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_scatter(paired: pd.DataFrame, out_path: Path) -> None:
    providers = sorted(paired["provider"].unique())
    fig, axes = plt.subplots(1, len(providers), figsize=(5 * len(providers), 5))
    if len(providers) == 1:
        axes = [axes]
    for ax, provider in zip(axes, providers):
        sub = paired[paired["provider"] == provider]
        for gate, grp in sub.groupby("gate"):
            ax.scatter(grp["p_text"], grp["p_image"], s=14, alpha=0.5, label=gate)
        ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
        ax.set_xlabel("text p(yes)")
        ax.set_ylabel("image p(yes)")
        ax.set_title(provider)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fmt_p(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


def analyze_gates_run(config: GatesConfig, run_dir: Path) -> Path:
    """Produce CSVs, figures, and a markdown summary for a gates run."""
    analysis_dir = run_dir / "analysis"
    figures_dir = analysis_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cal_frames, vis_frames, sel_frames = [], [], []
    for jsonl in sorted((run_dir / "calibration").glob("*.jsonl")):
        cal_frames.append(load_records(jsonl))
    for jsonl in sorted((run_dir / "vision").glob("*.jsonl")):
        vis_frames.append(load_records(jsonl))
    for csv in sorted((run_dir / "selection").glob("selected_*.csv")):
        sel_frames.append(pd.read_csv(csv))
    if not cal_frames or not vis_frames or not sel_frames:
        raise FileNotFoundError(
            f"Missing calibration/vision/selection outputs under {run_dir}"
        )
    calibration_df = pd.concat(cal_frames, ignore_index=True)
    vision_df = pd.concat(vis_frames, ignore_index=True)
    selected_df = pd.concat(sel_frames, ignore_index=True)

    paired = build_paired_table(calibration_df, vision_df, selected_df)
    paired.to_csv(analysis_dir / "paired_items.csv", index=False)

    # Boundary quality of the selected sets
    sel_quality = (
        selected_df.groupby(["provider", "gate"])
        .agg(
            n_items=("item_id", "size"),
            mean_p_text=("p_yes", "mean"),
            min_p_text=("p_yes", "min"),
            max_p_text=("p_yes", "max"),
            n_in_band=("in_band", "sum"),
        )
        .reset_index()
    )
    sel_quality.to_csv(analysis_dir / "selection_quality.csv", index=False)

    # Effects per variant (pooled over gates, and per gate)
    effects_pooled = _effect_rows(paired, ["provider", "variant_id"])
    effects_pooled.to_csv(analysis_dir / "variant_effects_pooled.csv", index=False)
    effects_by_gate = _effect_rows(paired, ["provider", "gate", "variant_id"])
    effects_by_gate.to_csv(analysis_dir / "variant_effects_by_gate.csv", index=False)

    # Modality effect: reference variants only
    std = paired[paired["variant_id"].isin(["T1_serif", REFERENCE_VARIANT])]
    modality = _effect_rows(std, ["provider", "gate"])
    modality.to_csv(analysis_dir / "modality_effect.csv", index=False)

    # OpenDyslexic fairness contrast
    od_summary, od_items = opendyslexic_contrast(vision_df)
    od_summary.to_csv(analysis_dir / "opendyslexic_contrast.csv", index=False)
    od_items.to_csv(analysis_dir / "opendyslexic_items.csv", index=False)

    # Figures
    _plot_forest(
        effects_pooled,
        figures_dir / "forest_variants_pooled.png",
        "Gate decision shift by visual variant (pooled over gates)",
    )
    for gate in paired["gate"].unique():
        _plot_forest(
            effects_by_gate[effects_by_gate["gate"] == gate],
            figures_dir / f"forest_variants_{gate}.png",
            f"Gate decision shift by visual variant ({gate})",
        )
    _plot_scatter(paired, figures_dir / "scatter_text_vs_image.png")

    # Markdown summary
    lines: List[str] = ["# Gates run summary\n"]
    lines.append("## Boundary calibration quality (selected items)\n")
    lines.append(sel_quality.to_markdown(index=False))
    lines.append("\n\n## Modality effect at the boundary (reference fonts only)\n")
    lines.append(
        modality[
            ["provider", "gate", "n_items", "mean_delta_p", "mean_delta_fav",
             "ci_lo_fav", "ci_hi_fav", "sign_test_p", "maj_flip_rate", "maj_net_fav"]
        ].to_markdown(index=False)
    )
    lines.append("\n\n## Variant effects pooled over gates\n")
    lines.append(
        effects_pooled[
            ["provider", "variant_id", "n_items", "mean_delta_fav",
             "ci_lo_fav", "ci_hi_fav", "sign_test_p", "maj_flip_rate", "maj_net_fav"]
        ].to_markdown(index=False)
    )
    if not od_summary.empty:
        lines.append("\n\n## OpenDyslexic vs sans reference (within-image)\n")
        lines.append(
            od_summary[
                ["provider", "gate", "n_items", "mean_delta_fav",
                 "ci_lo_fav", "ci_hi_fav", "sign_test_p"]
            ].to_markdown(index=False)
        )
    lines.append("\n")
    (analysis_dir / "summary.md").write_text("\n".join(lines))

    logger.info(f"Analysis written to {analysis_dir}")
    return analysis_dir
