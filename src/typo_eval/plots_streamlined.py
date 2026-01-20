"""Streamlined plotting functions for analysis figures."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typo_eval.taxonomy import get_variant_metadata

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def plot_flip_by_dimension(flip_rates: pd.DataFrame, output_path: Path) -> None:
    """
    Figure 1: Horizontal bar chart showing flip rate by dimension.
    Sorted descending by flip rate. Include error bars for 95% CI.
    """
    dim_data = flip_rates[flip_rates["group_type"] == "dimension"].copy()
    if len(dim_data) == 0:
        logger.warning("No dimension data to plot")
        return

    dim_data = dim_data.sort_values("flip_rate", ascending=True)  # For horizontal bars

    fig, ax = plt.subplots(figsize=(8, 6))

    y_pos = np.arange(len(dim_data))
    bars = ax.barh(y_pos, dim_data["flip_rate"], color="steelblue", alpha=0.8)

    # Error bars
    xerr_low = dim_data["flip_rate"] - dim_data["ci_low"]
    xerr_high = dim_data["ci_high"] - dim_data["flip_rate"]
    ax.errorbar(
        dim_data["flip_rate"],
        y_pos,
        xerr=[xerr_low, xerr_high],
        fmt="none",
        color="black",
        capsize=3,
        linewidth=1,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(dim_data["group_id"])
    ax.set_xlabel("Flip Rate (Vision vs Text)", fontsize=11)
    ax.set_title("Which Dimensions Are Most Affected?", fontsize=13, fontweight="bold")
    ax.set_xlim(0, min(0.6, dim_data["flip_rate"].max() * 1.15))

    # Add value labels
    for i, (idx, row) in enumerate(dim_data.iterrows()):
        rate = row["flip_rate"]
        ci_h = row["ci_high"]
        ax.text(ci_h + 0.01, i, f"{rate:.1%}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure to {output_path}")


def plot_flip_by_variant(flip_rates: pd.DataFrame, output_path: Path) -> None:
    """
    Figure 2: Horizontal bar chart showing flip rate by typography variant.
    Color-code by variant bucket (font_family, emphasis, capitalization).
    """
    var_data = flip_rates[flip_rates["group_type"] == "variant"].copy()
    if len(var_data) == 0:
        logger.warning("No variant data to plot")
        return

    # Add variant_bucket metadata
    var_data["variant_bucket"] = var_data["group_id"].apply(
        lambda vid: (
            get_variant_metadata(vid).variant_bucket
            if get_variant_metadata(vid)
            else "unknown"
        )
    )

    var_data = var_data.sort_values("flip_rate", ascending=True)

    # Color mapping for variant buckets
    color_map = {
        "font_family": "steelblue",
        "emphasis": "coral",
        "capitalization": "seagreen",
        "unknown": "gray",
    }
    colors = [color_map.get(bucket, "gray") for bucket in var_data["variant_bucket"]]

    fig, ax = plt.subplots(figsize=(8, max(5, len(var_data) * 0.35)))

    y_pos = np.arange(len(var_data))
    bars = ax.barh(y_pos, var_data["flip_rate"], color=colors, alpha=0.8)

    # Error bars
    xerr_low = var_data["flip_rate"] - var_data["ci_low"]
    xerr_high = var_data["ci_high"] - var_data["flip_rate"]
    ax.errorbar(
        var_data["flip_rate"],
        y_pos,
        xerr=[xerr_low, xerr_high],
        fmt="none",
        color="black",
        capsize=3,
        linewidth=1,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(var_data["group_id"])
    ax.set_xlabel("Flip Rate (Vision vs Text)", fontsize=11)
    ax.set_title(
        "Which Typography Variants Cause Most Disagreement?",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(0, min(0.3, var_data["flip_rate"].max() * 1.15))

    # Add value labels
    for i, (idx, row) in enumerate(var_data.iterrows()):
        rate = row["flip_rate"]
        ci_h = row["ci_high"]
        ax.text(ci_h + 0.005, i, f"{rate:.1%}", va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_map["font_family"], label="Font Family"),
        Patch(facecolor=color_map["emphasis"], label="Emphasis"),
        Patch(facecolor=color_map["capitalization"], label="Capitalization"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure to {output_path}")


def plot_direction_by_dimension(bias_direction: pd.DataFrame, output_path: Path) -> None:
    """
    Figure 3: Diverging horizontal bar chart showing direction bias by dimension.
    Center at 50%. Left = YES→NO (red), Right = NO→YES (blue).
    """
    dim_data = bias_direction[bias_direction["group_type"] == "dimension"].copy()
    if len(dim_data) == 0:
        logger.warning("No dimension direction data to plot")
        return

    dim_data = dim_data.sort_values("total_flips", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(dim_data))

    # Plot as diverging from center (50%)
    # Negative = YES→NO (toward rejection), Positive = NO→YES (toward approval)
    left_bars = dim_data["pct_no_to_yes"] - 50  # NO→YES
    right_bars = -(dim_data["pct_yes_to_no"] - 50)  # YES→NO (negative value)

    ax.barh(y_pos, left_bars, left=50, color="steelblue", alpha=0.8, label="NO→YES (Vision more positive)")
    ax.barh(y_pos, right_bars, left=50, color="coral", alpha=0.8, label="YES→NO (Vision more negative)")

    ax.axvline(x=50, color="black", linestyle="-", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dim_data["group_id"])
    ax.set_xlabel("Direction of Disagreement (%)", fontsize=11)
    ax.set_title(
        "When Vision and Text Disagree, Which Way?",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)

    # Add grid
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure to {output_path}")


def plot_direction_by_variant(bias_direction: pd.DataFrame, output_path: Path) -> None:
    """
    Figure 4: Diverging horizontal bar chart showing direction bias by variant.
    Similar to dimension plot but for typography variants.
    """
    var_data = bias_direction[bias_direction["group_type"] == "variant"].copy()
    if len(var_data) == 0:
        logger.warning("No variant direction data to plot")
        return

    # Add variant_bucket metadata
    var_data["variant_bucket"] = var_data["group_id"].apply(
        lambda vid: (
            get_variant_metadata(vid).variant_bucket
            if get_variant_metadata(vid)
            else "unknown"
        )
    )

    var_data = var_data.sort_values("total_flips", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(var_data) * 0.35)))

    y_pos = np.arange(len(var_data))

    # Plot as diverging from center (50%)
    left_bars = var_data["pct_no_to_yes"] - 50  # NO→YES
    right_bars = -(var_data["pct_yes_to_no"] - 50)  # YES→NO

    ax.barh(y_pos, left_bars, left=50, color="steelblue", alpha=0.8, label="NO→YES")
    ax.barh(y_pos, right_bars, left=50, color="coral", alpha=0.8, label="YES→NO")

    ax.axvline(x=50, color="black", linestyle="-", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(var_data["group_id"])
    ax.set_xlabel("Direction of Disagreement (%)", fontsize=11)
    ax.set_title(
        "Typography Direction Bias",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)

    # Add grid
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure to {output_path}")


def plot_decision_flip(decision_analysis: pd.DataFrame, provider: str, output_path: Path) -> None:
    """
    Figure 5: Bar chart showing decision flip rate.
    Annotate with direction (% toward NO).
    """
    # Get flip rate data
    flip_data = decision_analysis[
        (decision_analysis["metric"] == "flip_rate") &
        (decision_analysis["group_type"] == "overall")
    ]

    if len(flip_data) == 0:
        logger.warning("No decision flip rate data to plot")
        return

    # Get direction data
    direction_data = decision_analysis[
        (decision_analysis["metric"] == "direction") &
        (decision_analysis["group_type"] == "overall")
    ]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Single bar for overall flip rate
    flip_rate = flip_data.iloc[0]["value"]
    ci_low = flip_data.iloc[0]["ci_low"]
    ci_high = flip_data.iloc[0]["ci_high"]

    bar = ax.bar([0], [flip_rate], color="steelblue", alpha=0.8, width=0.5)

    # Error bar
    if not pd.isna(ci_low) and not pd.isna(ci_high):
        ax.errorbar(
            [0],
            [flip_rate],
            yerr=[[flip_rate - ci_low], [ci_high - flip_rate]],
            fmt="none",
            color="black",
            capsize=5,
            linewidth=2,
        )

    ax.set_xticks([0])
    ax.set_xticklabels([provider.capitalize()])
    ax.set_ylabel("Decision Flip Rate", fontsize=11)
    ax.set_title(
        f"How Often Do Decisions Flip?\n{provider.capitalize()}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0, min(0.3, flip_rate * 1.3))

    # Add value annotation
    ax.text(0, flip_rate + 0.01, f"{flip_rate:.1%}", ha="center", fontsize=12, fontweight="bold")

    # Add direction annotation
    if len(direction_data) > 0:
        net_bias = direction_data.iloc[0]["value"]
        if net_bias < -0.5:
            direction_text = f"Net bias: {net_bias:.0%}\n(Vision less likely to escalate)"
        elif net_bias > 0.5:
            direction_text = f"Net bias: {net_bias:+.0%}\n(Vision more likely to escalate)"
        else:
            direction_text = f"Net bias: {net_bias:+.0%}\n(Mixed)"

        ax.text(
            0,
            flip_rate * 0.5,
            direction_text,
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure to {output_path}")
