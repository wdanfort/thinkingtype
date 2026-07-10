"""Streamlined analysis functions for consolidated outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typo_eval.stats import (
    bh_fdr,
    cluster_bootstrap_ci,
    lift_stat,
    mcnemar_exact,
    net_bias_stat,
    per_cluster_counts,
    rate_stat,
)
from typo_eval.taxonomy import get_variant_metadata

logger = logging.getLogger(__name__)


def generate_flip_rates_csv(
    paired: pd.DataFrame,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate consolidated flip rates CSV with bootstrap CIs.

    Output schema:
    - group_type: "overall" | "dimension" | "variant"
    - group_id: specific ID or "all"
    - flip_count: number of flips
    - total: total comparisons
    - flip_rate: proportion that flipped
    - ci_low: lower bound of 95% CI
    - ci_high: upper bound of 95% CI
    """
    # Filter out T8_small_text variant
    paired = paired[paired["variant_id"] != "T8_small_text"].copy()
    paired["_one"] = 1

    rows = []

    # Cluster bootstrap by sentence, resampling clusters with multiplicity
    def compute_ci(group_df):
        counts = per_cluster_counts(group_df, "sentence_id", ["flip", "_one"])
        return cluster_bootstrap_ci(counts, rate_stat, n_boot=n_boot, alpha=alpha, seed=seed)

    # Overall flip rate
    flip_count = int(paired["flip"].sum())
    total = len(paired)
    flip_rate = paired["flip"].mean()
    ci_low, ci_high = compute_ci(paired)

    rows.append({
        "group_type": "overall",
        "group_id": "all",
        "flip_count": flip_count,
        "total": total,
        "flip_rate": flip_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    })

    # By dimension
    if "dimension_id" in paired.columns:
        for dim_id, dim_df in paired.groupby("dimension_id"):
            flip_count = int(dim_df["flip"].sum())
            total = len(dim_df)
            flip_rate = dim_df["flip"].mean()
            ci_low, ci_high = compute_ci(dim_df)

            rows.append({
                "group_type": "dimension",
                "group_id": dim_id,
                "flip_count": flip_count,
                "total": total,
                "flip_rate": flip_rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            })

    # By variant
    for var_id, var_df in paired.groupby("variant_id"):
        flip_count = int(var_df["flip"].sum())
        total = len(var_df)
        flip_rate = var_df["flip"].mean()
        ci_low, ci_high = compute_ci(var_df)

        rows.append({
            "group_type": "variant",
            "group_id": var_id,
            "flip_count": flip_count,
            "total": total,
            "flip_rate": flip_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    df = pd.DataFrame(rows)

    # Sort: overall first, then dimensions by flip_rate desc, then variants by flip_rate desc
    df["sort_order"] = df["group_type"].map({"overall": 0, "dimension": 1, "variant": 2})
    df = df.sort_values(["sort_order", "flip_rate"], ascending=[True, False])
    df = df.drop(columns=["sort_order"])

    return df


def generate_bias_direction_csv(
    paired: pd.DataFrame,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate consolidated bias direction CSV with inference columns.

    Output schema:
    - group_type: "overall" | "dimension" | "variant"
    - group_id: specific ID or "all"
    - no_to_yes: count of NO→YES flips
    - yes_to_no: count of YES→NO flips
    - total_flips: total flips
    - pct_no_to_yes: percentage NO→YES
    - pct_yes_to_no: percentage YES→NO
    - net_bias: pct_no_to_yes - pct_yes_to_no (positive = vision more positive)
    - net_bias_ci_low / net_bias_ci_high: cluster-bootstrap 95% CI for net_bias
      (resampling sentences with replacement; primary inference)
    - p_mcnemar: exact McNemar p-value on discordant pairs (secondary;
      anti-conservative because pairs within a sentence are correlated)
    - p_bh: Benjamini-Hochberg adjusted p_mcnemar. The correction family is
      all rows of this table (overall + per-dimension + per-variant) within
      a single run.
    """
    # Filter out T8_small_text variant
    paired = paired[paired["variant_id"] != "T8_small_text"].copy()
    paired["_n01"] = (paired["delta"] == 1).astype(int)
    paired["_n10"] = (paired["delta"] == -1).astype(int)

    rows = []

    def compute_row(group_type, group_id, group_df):
        no_to_yes = int(group_df["_n01"].sum())
        yes_to_no = int(group_df["_n10"].sum())
        total_flips = no_to_yes + yes_to_no
        if total_flips > 0:
            pct_no_to_yes = 100 * no_to_yes / total_flips
            pct_yes_to_no = 100 * yes_to_no / total_flips
            net_bias = pct_no_to_yes - pct_yes_to_no
        else:
            pct_no_to_yes = pct_yes_to_no = net_bias = 0

        counts = per_cluster_counts(group_df, "sentence_id", ["_n01", "_n10"])
        ci_low, ci_high = cluster_bootstrap_ci(
            counts, net_bias_stat, n_boot=n_boot, alpha=alpha, seed=seed
        )

        return {
            "group_type": group_type,
            "group_id": group_id,
            "no_to_yes": no_to_yes,
            "yes_to_no": yes_to_no,
            "total_flips": total_flips,
            "pct_no_to_yes": pct_no_to_yes,
            "pct_yes_to_no": pct_yes_to_no,
            "net_bias": net_bias,
            "net_bias_ci_low": ci_low,
            "net_bias_ci_high": ci_high,
            "p_mcnemar": mcnemar_exact(no_to_yes, yes_to_no),
        }

    # Overall
    rows.append(compute_row("overall", "all", paired))

    # By dimension
    if "dimension_id" in paired.columns:
        for dim_id, dim_df in paired.groupby("dimension_id"):
            rows.append(compute_row("dimension", dim_id, dim_df))

    # By variant
    for var_id, var_df in paired.groupby("variant_id"):
        rows.append(compute_row("variant", var_id, var_df))

    df = pd.DataFrame(rows)

    # BH-FDR across all rows of this table (documented family choice)
    df["p_bh"] = bh_fdr(df["p_mcnemar"].to_numpy())

    # Sort by total_flips descending within each group_type
    df["sort_order"] = df["group_type"].map({"overall": 0, "dimension": 1, "variant": 2})
    df = df.sort_values(["sort_order", "total_flips"], ascending=[True, False])
    df = df.drop(columns=["sort_order"])

    return df


def generate_decision_analysis_csv(
    decision_paired: pd.DataFrame,
    dimension_paired: pd.DataFrame,
    provider: str,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate decision-specific analysis CSV.

    Includes:
    - Decision flip rates (overall and by variant)
    - Direction bias
    - Correlation between dimension flip rate and decision flip rate
    - Lift: how much more likely a dimension flip predicts a decision flip
    """
    # Filter out T8_small_text variant
    if decision_paired is not None and len(decision_paired) > 0:
        decision_paired = decision_paired[decision_paired["variant_id"] != "T8_small_text"].copy()
    if dimension_paired is not None and len(dimension_paired) > 0:
        dimension_paired = dimension_paired[dimension_paired["variant_id"] != "T8_small_text"].copy()

    rows = []

    if len(decision_paired) == 0:
        return pd.DataFrame(rows)

    decision_paired["_one"] = 1
    decision_paired["_n01"] = (decision_paired["delta"] == 1).astype(int)
    decision_paired["_n10"] = (decision_paired["delta"] == -1).astype(int)

    # Overall decision flip rate with cluster-bootstrap CI
    flip_rate = decision_paired["flip"].mean()
    counts = per_cluster_counts(decision_paired, "sentence_id", ["flip", "_one"])
    ci_low, ci_high = cluster_bootstrap_ci(counts, rate_stat, n_boot=n_boot, alpha=alpha, seed=seed)

    rows.append({
        "metric": "flip_rate",
        "group_type": "overall",
        "group_id": "all",
        "value": flip_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": len(decision_paired),
    })

    # Flip rate by variant, with cluster-bootstrap CIs
    for var_id, var_df in decision_paired.groupby("variant_id"):
        flip_rate = var_df["flip"].mean()
        var_counts = per_cluster_counts(var_df, "sentence_id", ["flip", "_one"])
        var_ci_low, var_ci_high = cluster_bootstrap_ci(
            var_counts, rate_stat, n_boot=n_boot, alpha=alpha, seed=seed
        )

        rows.append({
            "metric": "flip_rate",
            "group_type": "variant",
            "group_id": var_id,
            "value": flip_rate,
            "ci_low": var_ci_low,
            "ci_high": var_ci_high,
            "n": len(var_df),
        })

    # Direction bias with cluster-bootstrap CI and exact McNemar p-value
    no_to_yes = int(decision_paired["_n01"].sum())
    yes_to_no = int(decision_paired["_n10"].sum())
    n_flipped = no_to_yes + yes_to_no
    net_bias = (no_to_yes - yes_to_no) / n_flipped if n_flipped > 0 else 0

    dir_counts = per_cluster_counts(decision_paired, "sentence_id", ["_n01", "_n10"])
    # net_bias here is a fraction (not percent), so scale the percent-based stat
    dir_ci_low, dir_ci_high = cluster_bootstrap_ci(
        dir_counts,
        lambda s: net_bias_stat(s) / 100.0,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )

    rows.append({
        "metric": "direction",
        "group_type": "overall",
        "group_id": "all",
        "value": net_bias,
        "ci_low": dir_ci_low,
        "ci_high": dir_ci_high,
        "n": n_flipped,
    })

    rows.append({
        "metric": "direction_p_mcnemar",
        "group_type": "overall",
        "group_id": "all",
        "value": mcnemar_exact(no_to_yes, yes_to_no),
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n": n_flipped,
    })

    # Compute correlations and lift (only if we have dimension data)
    if len(dimension_paired) > 0 and "dimension_id" in dimension_paired.columns:
        # For each dimension, compute:
        # 1. Correlation: does high flip rate in this dimension predict decision flips?
        # 2. Lift: P(decision flip | dimension flip) / P(decision flip | no dimension flip)

        # Merge dimension and decision data by sentence_id and variant_id
        dim_agg = (
            dimension_paired.groupby(["sentence_id", "variant_id", "dimension_id"])
            .agg({"flip": "mean"})
            .reset_index()
            .rename(columns={"flip": "dim_flip"})
        )

        dec_agg = (
            decision_paired.groupby(["sentence_id", "variant_id"])
            .agg({"flip": "mean"})
            .reset_index()
            .rename(columns={"flip": "dec_flip"})
        )

        for dim_id in dimension_paired["dimension_id"].unique():
            dim_for_this = dim_agg[dim_agg["dimension_id"] == dim_id]
            merged = dim_for_this.merge(dec_agg, on=["sentence_id", "variant_id"], how="inner")

            if len(merged) > 0:
                # Correlation
                if merged["dim_flip"].std() > 0 and merged["dec_flip"].std() > 0:
                    corr = merged["dim_flip"].corr(merged["dec_flip"])
                else:
                    corr = 0

                # Lift with cluster-bootstrap CI. Per-pair indicators:
                # a = dim flip & dec flip, b = dim flip & no dec flip,
                # c = no dim flip & dec flip, d = neither.
                merged = merged.copy()
                dim_flipped = merged["dim_flip"] > 0.5
                dec_flipped = merged["dec_flip"] > 0.5
                merged["_a"] = (dim_flipped & dec_flipped).astype(int)
                merged["_b"] = (dim_flipped & ~dec_flipped).astype(int)
                merged["_c"] = (~dim_flipped & dec_flipped).astype(int)
                merged["_d"] = (~dim_flipped & ~dec_flipped).astype(int)

                lift_counts = per_cluster_counts(
                    merged, "sentence_id", ["_a", "_b", "_c", "_d"]
                )
                lift = lift_stat(lift_counts.sum(axis=0))
                if np.isnan(lift):
                    lift = 0
                    lift_ci_low, lift_ci_high = np.nan, np.nan
                else:
                    lift_ci_low, lift_ci_high = cluster_bootstrap_ci(
                        lift_counts, lift_stat, n_boot=n_boot, alpha=alpha, seed=seed
                    )

                rows.append({
                    "metric": "correlation",
                    "group_type": "dimension",
                    "group_id": dim_id,
                    "value": corr,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n": len(merged),
                })

                rows.append({
                    "metric": "lift",
                    "group_type": "dimension",
                    "group_id": dim_id,
                    "value": lift,
                    "ci_low": lift_ci_low,
                    "ci_high": lift_ci_high,
                    "n": len(merged),
                })

    return pd.DataFrame(rows)


def generate_headline_metrics_json(
    paired: pd.DataFrame,
    decision_paired: pd.DataFrame,
    flip_rates: pd.DataFrame,
    bias_direction: pd.DataFrame,
    decision_analysis: pd.DataFrame,
    config,
    run_id: str,
    provider: str,
    model: str,
) -> dict:
    """
    Generate headline metrics JSON for programmatic access.
    """
    # Overall dimension metrics
    overall_flip = flip_rates[flip_rates["group_type"] == "overall"].iloc[0]

    # Top flipping dimension
    dim_flips = flip_rates[flip_rates["group_type"] == "dimension"]
    if len(dim_flips) > 0:
        top_dim = dim_flips.iloc[0]
        top_dim_id = top_dim["group_id"]
        top_dim_rate = top_dim["flip_rate"]
    else:
        top_dim_id = None
        top_dim_rate = None

    # Top flipping variant
    var_flips = flip_rates[flip_rates["group_type"] == "variant"]
    if len(var_flips) > 0:
        top_var = var_flips.iloc[0]
        top_var_id = top_var["group_id"]
        top_var_rate = top_var["flip_rate"]
    else:
        top_var_id = None
        top_var_rate = None

    # Decision metrics
    if len(decision_analysis) > 0:
        dec_flip_overall = decision_analysis[
            (decision_analysis["metric"] == "flip_rate") &
            (decision_analysis["group_type"] == "overall")
        ]
        dec_direction_overall = decision_analysis[
            (decision_analysis["metric"] == "direction") &
            (decision_analysis["group_type"] == "overall")
        ]

        if len(dec_flip_overall) > 0:
            dec_flip_rate = float(dec_flip_overall.iloc[0]["value"])
            dec_flip_ci = [
                float(dec_flip_overall.iloc[0]["ci_low"]),
                float(dec_flip_overall.iloc[0]["ci_high"])
            ]
            dec_n = int(dec_flip_overall.iloc[0]["n"])
        else:
            dec_flip_rate = None
            dec_flip_ci = None
            dec_n = 0

        if len(dec_direction_overall) > 0:
            dec_net_bias = float(dec_direction_overall.iloc[0]["value"])
        else:
            dec_net_bias = None

        # Top correlations
        correlations = decision_analysis[decision_analysis["metric"] == "correlation"]
        if len(correlations) > 0:
            top_corr = correlations.nlargest(3, "value")
            corr_dict = {}
            for _, row in top_corr.iterrows():
                dim_id = row["group_id"]
                # Find corresponding lift
                lift_row = decision_analysis[
                    (decision_analysis["metric"] == "lift") &
                    (decision_analysis["group_id"] == dim_id)
                ]
                if len(lift_row) > 0:
                    corr_dict[f"{dim_id}_decision_lift"] = float(lift_row.iloc[0]["value"])
        else:
            corr_dict = {}
    else:
        dec_flip_rate = None
        dec_flip_ci = None
        dec_n = 0
        dec_net_bias = None
        corr_dict = {}

    return {
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "timestamp": pd.Timestamp.now().isoformat(),

        "dimension_analysis": {
            "overall_flip_rate": float(overall_flip["flip_rate"]),
            "overall_ci": [float(overall_flip["ci_low"]), float(overall_flip["ci_high"])],
            "n_comparisons": int(overall_flip["total"]),
            "top_flipping_dimension": top_dim_id,
            "top_flipping_dimension_rate": float(top_dim_rate) if top_dim_rate else None,
            "top_flipping_variant": top_var_id,
            "top_flipping_variant_rate": float(top_var_rate) if top_var_rate else None,
        },

        "decision_analysis": {
            "flip_rate": dec_flip_rate,
            "flip_ci": dec_flip_ci,
            "n_decisions": dec_n,
            "direction_net_bias": dec_net_bias,
            "direction_interpretation": "Vision less likely to escalate" if dec_net_bias and dec_net_bias < -0.5 else "Mixed" if dec_net_bias else None,
        },

        "key_correlations": corr_dict,
    }


def generate_summary_md_streamlined(
    flip_rates: pd.DataFrame,
    bias_direction: pd.DataFrame,
    decision_analysis: pd.DataFrame,
    headline_metrics: dict,
    output_path: Path,
) -> None:
    """
    Generate streamlined summary.md file.
    """
    lines = []

    lines.append("# Typography Evaluation Analysis Summary")
    lines.append("")
    lines.append(f"**Run ID:** {headline_metrics['run_id']}")
    lines.append(f"**Provider:** {headline_metrics['provider']}")
    lines.append(f"**Model:** {headline_metrics['model']}")
    lines.append("")

    # Dimension analysis
    dim_metrics = headline_metrics["dimension_analysis"]
    lines.append("## Dimension Analysis")
    lines.append("")
    lines.append(f"- **Overall Flip Rate:** {dim_metrics['overall_flip_rate']:.1%} "
                 f"(95% CI: [{dim_metrics['overall_ci'][0]:.1%}, {dim_metrics['overall_ci'][1]:.1%}])")
    lines.append(f"- **Total Comparisons:** {dim_metrics['n_comparisons']}")
    lines.append(f"- **Top Flipping Dimension:** {dim_metrics['top_flipping_dimension']} "
                 f"({dim_metrics['top_flipping_dimension_rate']:.1%})")
    lines.append(f"- **Top Flipping Variant:** {dim_metrics['top_flipping_variant']} "
                 f"({dim_metrics['top_flipping_variant_rate']:.1%})")
    lines.append("")

    # Top dimensions table
    lines.append("### Top Dimensions by Flip Rate")
    lines.append("")
    lines.append("| Dimension | Flip Rate | 95% CI | Total |")
    lines.append("|-----------|-----------|--------|-------|")

    dim_data = flip_rates[flip_rates["group_type"] == "dimension"].head(5)
    for _, row in dim_data.iterrows():
        lines.append(
            f"| {row['group_id']} | {row['flip_rate']:.1%} | "
            f"[{row['ci_low']:.1%}, {row['ci_high']:.1%}] | {row['total']} |"
        )
    lines.append("")

    # Direction bias
    lines.append("### Direction Bias (Top Dimensions)")
    lines.append("")
    lines.append("| Dimension | Total Flips | NO→YES | YES→NO | Net Bias |")
    lines.append("|-----------|-------------|---------|---------|----------|")

    dir_data = bias_direction[bias_direction["group_type"] == "dimension"]
    dir_data = dir_data.sort_values("total_flips", ascending=False).head(5)
    for _, row in dir_data.iterrows():
        bias_str = f"+{row['net_bias']:.0f}%" if row['net_bias'] > 0 else f"{row['net_bias']:.0f}%"
        lines.append(
            f"| {row['group_id']} | {row['total_flips']} | "
            f"{row['pct_no_to_yes']:.0f}% | {row['pct_yes_to_no']:.0f}% | {bias_str} |"
        )
    lines.append("")

    # Decision analysis
    if len(decision_analysis) > 0:
        dec_metrics = headline_metrics["decision_analysis"]
        lines.append("## Decision Analysis")
        lines.append("")

        if dec_metrics["flip_rate"] is not None:
            lines.append(f"- **Decision Flip Rate:** {dec_metrics['flip_rate']:.1%} "
                         f"(95% CI: [{dec_metrics['flip_ci'][0]:.1%}, {dec_metrics['flip_ci'][1]:.1%}])")
            lines.append(f"- **Total Decisions:** {dec_metrics['n_decisions']}")

        if dec_metrics["direction_net_bias"] is not None:
            lines.append(f"- **Direction Net Bias:** {dec_metrics['direction_net_bias']:+.1%}")
            lines.append(f"- **Interpretation:** {dec_metrics['direction_interpretation']}")

        lines.append("")

        # Key correlations
        if headline_metrics["key_correlations"]:
            lines.append("### Key Dimension-Decision Correlations")
            lines.append("")
            lines.append("| Dimension | Lift |")
            lines.append("|-----------|------|")

            for key, value in headline_metrics["key_correlations"].items():
                dim_name = key.replace("_decision_lift", "")
                lines.append(f"| {dim_name} | {value:.2f}x |")
            lines.append("")

    # File references
    lines.append("## Output Files")
    lines.append("")
    lines.append("**CSVs:**")
    lines.append("- `flip_rates.csv` - Flip rates by dimension and variant with bootstrap CIs")
    lines.append("- `bias_direction.csv` - Direction bias (NO→YES vs YES→NO)")
    lines.append("- `decision_analysis.csv` - Decision-specific metrics and correlations")
    lines.append("- `headline_metrics.json` - Key metrics for programmatic access")
    lines.append("")
    lines.append("**Figures:**")
    lines.append("- `fig1_flip_by_dimension.png` - Flip rates by dimension")
    lines.append("- `fig2_flip_by_variant.png` - Flip rates by typography variant")
    lines.append("- `fig3_direction_by_dimension.png` - Direction bias by dimension")
    lines.append("- `fig4_direction_by_variant.png` - Direction bias by variant")
    if len(decision_analysis) > 0:
        lines.append("- `fig5_decision_flip.png` - Decision flip rate and direction")
    lines.append("")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved summary to {output_path}")
