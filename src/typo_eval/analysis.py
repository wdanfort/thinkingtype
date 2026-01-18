"""Analysis and plotting utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typo_eval.config import TypoEvalConfig
from typo_eval.prompts import coerce_to_binary

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def load_results(jsonl_path: Path) -> pd.DataFrame:
    """Load results from JSONL file."""
    records = []
    with jsonl_path.open("r") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)


def build_delta_table(
    results: pd.DataFrame,
    sentences_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build paired comparison table between OCR baseline and image variants.

    Returns DataFrame with columns:
    - sentence_id/artifact_id
    - variant_id
    - dimension_id (for dimensions mode)
    - ocr_response (baseline)
    - image_response
    - delta (image - ocr)
    - flip (1 if different, 0 otherwise)
    - approval_ocr (YES rate for OCR baseline)
    - approval_image (YES rate for image variant)
    - flip_direction (NO→YES, YES→NO, or no_flip)
    """
    results = results.copy()

    # Coerce parsed_response to binary
    results["response_01"] = results["parsed_response"].apply(coerce_to_binary)
    results = results.dropna(subset=["response_01"]).copy()
    results["response_01"] = results["response_01"].astype(int)

    # Separate OCR and image results
    ocr_rows = results[results["representation"] == "ocr"]
    image_rows = results[results["representation"] == "image"]

    # Build OCR baseline (deduped)
    if "dimension_id" in results.columns and results["dimension_id"].notna().any():
        # Dimensions mode
        ocr_base = (
            ocr_rows.groupby(["sentence_id", "dimension_id"])["response_01"]
            .mean()
            .reset_index()
        )
        ocr_base["response_01"] = ocr_base["response_01"].round().astype(int)
        ocr_base = ocr_base.rename(columns={"response_01": "ocr"})

        image_agg = (
            image_rows.groupby(["sentence_id", "variant_id", "dimension_id"])["response_01"]
            .mean()
            .reset_index()
        )
        image_agg["response_01"] = image_agg["response_01"].round().astype(int)
        image_agg = image_agg.rename(columns={"response_01": "image"})

        paired = image_agg.merge(ocr_base, on=["sentence_id", "dimension_id"], how="left")
    else:
        # Decision mode (no dimension_id)
        ocr_base = (
            ocr_rows.groupby(["sentence_id"])["response_01"]
            .mean()
            .reset_index()
        )
        ocr_base["response_01"] = ocr_base["response_01"].round().astype(int)
        ocr_base = ocr_base.rename(columns={"response_01": "ocr"})

        image_agg = (
            image_rows.groupby(["sentence_id", "variant_id"])["response_01"]
            .mean()
            .reset_index()
        )
        image_agg["response_01"] = image_agg["response_01"].round().astype(int)
        image_agg = image_agg.rename(columns={"response_01": "image"})

        paired = image_agg.merge(ocr_base, on=["sentence_id"], how="left")

    # Filter rows with missing OCR baseline
    paired = paired.dropna(subset=["ocr"]).copy()
    paired["ocr"] = paired["ocr"].astype(int)

    # Compute delta and flip
    paired["delta"] = paired["image"] - paired["ocr"]
    paired["abs_delta"] = paired["delta"].abs()
    paired["flip"] = (paired["image"] != paired["ocr"]).astype(int)

    # Add approval rate columns (YES = 1, NO = 0)
    paired["approval_ocr"] = paired["ocr"]
    paired["approval_image"] = paired["image"]

    # Add flip directionality
    paired["flip_direction"] = paired["delta"].apply(
        lambda x: "NO→YES" if x == 1 else "YES→NO" if x == -1 else "no_flip"
    )

    # Add category from sentences_df if available
    if sentences_df is not None and "category" in sentences_df.columns:
        meta_unique = sentences_df.drop_duplicates(subset=["sentence_id"])
        paired = paired.merge(
            meta_unique[["sentence_id", "category"]],
            on="sentence_id",
            how="left",
        )

    return paired


def compute_flip_rates(paired: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute flip rates by variant, sentence, and optionally dimension/artifact type.

    Returns dict of DataFrames.
    """
    results = {}

    # Overall flip rate
    overall = pd.DataFrame([{
        "flip_rate": paired["flip"].mean(),
        "n": len(paired),
    }])
    results["overall"] = overall

    # By variant
    by_variant = (
        paired.groupby("variant_id")
        .agg(flip_rate=("flip", "mean"), n=("flip", "size"))
        .reset_index()
        .sort_values("flip_rate", ascending=False)
    )
    results["by_variant"] = by_variant

    # By sentence
    by_sentence = (
        paired.groupby("sentence_id")
        .agg(flip_rate=("flip", "mean"), n=("flip", "size"))
        .reset_index()
        .sort_values("flip_rate", ascending=False)
    )
    results["by_sentence"] = by_sentence

    # By dimension (if available)
    if "dimension_id" in paired.columns:
        by_dimension = (
            paired.groupby("dimension_id")
            .agg(flip_rate=("flip", "mean"), n=("flip", "size"))
            .reset_index()
            .sort_values("flip_rate", ascending=False)
        )
        results["by_dimension"] = by_dimension

    # By category (if available)
    if "category" in paired.columns and paired["category"].notna().any():
        by_category = (
            paired.groupby("category")
            .agg(flip_rate=("flip", "mean"), n=("flip", "size"))
            .reset_index()
            .sort_values("flip_rate", ascending=False)
        )
        results["by_category"] = by_category

    return results


def compute_approval_rates(paired: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute approval rates (% YES judgments) by variant, sentence, dimension, and category.

    Returns dict of DataFrames with approval_rate_ocr, approval_rate_image, and n.
    """
    results = {}

    # Overall approval rate
    overall = pd.DataFrame([{
        "approval_rate_ocr": paired["approval_ocr"].mean(),
        "approval_rate_image": paired["approval_image"].mean(),
        "n": len(paired),
    }])
    results["overall"] = overall

    # By variant
    by_variant = (
        paired.groupby("variant_id")
        .agg(
            approval_rate_ocr=("approval_ocr", "mean"),
            approval_rate_image=("approval_image", "mean"),
            n=("approval_ocr", "size")
        )
        .reset_index()
        .sort_values("approval_rate_image", ascending=False)
    )
    results["by_variant"] = by_variant

    # By sentence
    by_sentence = (
        paired.groupby("sentence_id")
        .agg(
            approval_rate_ocr=("approval_ocr", "mean"),
            approval_rate_image=("approval_image", "mean"),
            n=("approval_ocr", "size")
        )
        .reset_index()
        .sort_values("approval_rate_image", ascending=False)
    )
    results["by_sentence"] = by_sentence

    # By dimension (if available)
    if "dimension_id" in paired.columns:
        by_dimension = (
            paired.groupby("dimension_id")
            .agg(
                approval_rate_ocr=("approval_ocr", "mean"),
                approval_rate_image=("approval_image", "mean"),
                n=("approval_ocr", "size")
            )
            .reset_index()
            .sort_values("approval_rate_image", ascending=False)
        )
        results["by_dimension"] = by_dimension

    # By category (if available)
    if "category" in paired.columns and paired["category"].notna().any():
        by_category = (
            paired.groupby("category")
            .agg(
                approval_rate_ocr=("approval_ocr", "mean"),
                approval_rate_image=("approval_image", "mean"),
                n=("approval_ocr", "size")
            )
            .reset_index()
            .sort_values("approval_rate_image", ascending=False)
        )
        results["by_category"] = by_category

    return results


def compute_flip_directionality(paired: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute flip directionality (NO→YES vs YES→NO) breakdown.

    Returns dict of DataFrames with counts and rates for each direction.
    """
    results = {}

    # Filter to only flips
    flipped = paired[paired["flip"] == 1].copy()

    if len(flipped) == 0:
        # No flips, return empty DataFrames
        return results

    # Overall directionality
    overall = (
        flipped.groupby("flip_direction")
        .size()
        .reset_index(name="count")
    )
    overall["rate"] = overall["count"] / overall["count"].sum()
    results["overall"] = overall

    # By variant
    by_variant = (
        flipped.groupby(["variant_id", "flip_direction"])
        .size()
        .reset_index(name="count")
    )
    # Calculate rate within each variant
    by_variant["rate"] = by_variant.groupby("variant_id")["count"].transform(
        lambda x: x / x.sum()
    )
    results["by_variant"] = by_variant

    # By dimension (if available)
    if "dimension_id" in flipped.columns:
        by_dimension = (
            flipped.groupby(["dimension_id", "flip_direction"])
            .size()
            .reset_index(name="count")
        )
        by_dimension["rate"] = by_dimension.groupby("dimension_id")["count"].transform(
            lambda x: x / x.sum()
        )
        results["by_dimension"] = by_dimension

    # By category (if available)
    if "category" in flipped.columns and flipped["category"].notna().any():
        by_category = (
            flipped.groupby(["category", "flip_direction"])
            .size()
            .reset_index(name="count")
        )
        by_category["rate"] = by_category.groupby("category")["count"].transform(
            lambda x: x / x.sum()
        )
        results["by_category"] = by_category

    return results


def bootstrap_ci(
    paired: pd.DataFrame,
    column: str = "flip",
    group_col: Optional[str] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals by resampling sentences.

    Returns DataFrame with mean, ci_low, ci_high, n per group.
    """
    rng = np.random.default_rng(seed)
    sentences = paired["sentence_id"].unique()

    if group_col:
        groups = paired[group_col].unique()
        point_estimates = paired.groupby(group_col)[column].mean()
    else:
        groups = [None]
        point_estimates = pd.Series({"overall": paired[column].mean()})

    boot_results = {g: [] for g in groups}

    for _ in range(n_boot):
        # Resample sentences with replacement
        sample_sentences = rng.choice(sentences, size=len(sentences), replace=True)
        boot_df = paired[paired["sentence_id"].isin(sample_sentences)]

        if group_col:
            boot_means = boot_df.groupby(group_col)[column].mean()
            for g in groups:
                if g in boot_means.index:
                    boot_results[g].append(boot_means[g])
        else:
            boot_results[None].append(boot_df[column].mean())

    # Compute CIs
    rows = []
    for g, vals in boot_results.items():
        if not vals:
            continue
        vals = np.array(vals)
        lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])

        if group_col:
            mean_val = point_estimates.get(g, np.nan)
            n_val = len(paired[paired[group_col] == g])
        else:
            mean_val = point_estimates.iloc[0]
            n_val = len(paired)

        row = {
            "mean": float(mean_val),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "n": int(n_val),
        }
        if group_col:
            row[group_col] = g

        rows.append(row)

    return pd.DataFrame(rows)


def plot_flip_rate_by_variant(df: pd.DataFrame, output_path: Path) -> None:
    """Plot flip rate by variant as horizontal bar chart."""
    plt.figure(figsize=(10, max(5, len(df) * 0.3)))
    df_sorted = df.sort_values("flip_rate", ascending=True)
    plt.barh(df_sorted["variant_id"], df_sorted["flip_rate"], color="darkorange")
    plt.xlabel("Flip Rate")
    plt.title("Flip Rate vs OCR Baseline by Typography Variant")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_heatmap_sentence_variant(paired: pd.DataFrame, output_path: Path) -> None:
    """Plot heatmap of flips by sentence x variant."""
    pivot = paired.pivot_table(
        index="sentence_id",
        columns="variant_id",
        values="flip",
        aggfunc="mean",
        fill_value=0,
    )

    plt.figure(figsize=(12, max(6, len(pivot) * 0.25)))
    sns.heatmap(pivot, cmap="Reds", cbar=True, linewidths=0.3)
    plt.title("Flip vs OCR Baseline (Sentence x Variant)")
    plt.xlabel("Variant")
    plt.ylabel("Sentence ID")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_mean_delta_by_dimension(df: pd.DataFrame, output_path: Path) -> None:
    """Plot mean delta by dimension."""
    plt.figure(figsize=(10, 4.5))
    df_sorted = df.sort_values("mean")
    plt.bar(df_sorted["dimension_id"], df_sorted["mean"], color="steelblue")
    plt.axhline(0, color="gray", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Delta (Image - OCR)")
    plt.title("Mean Delta by Dimension")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_bootstrap_ci(df: pd.DataFrame, group_col: str, output_path: Path) -> None:
    """Plot bootstrap confidence intervals."""
    plt.figure(figsize=(10, 4.5))
    x = np.arange(len(df))
    y = df["mean"].to_numpy()
    yerr_low = y - df["ci_low"].to_numpy()
    yerr_high = df["ci_high"].to_numpy() - y

    plt.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="o", capsize=4, color="black")
    plt.axhline(0, color="gray", linewidth=1)
    plt.xticks(x, df[group_col], rotation=45, ha="right")
    plt.ylabel("Value")
    plt.title(f"Bootstrap 95% CI by {group_col}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_approval_rate_by_variant(df: pd.DataFrame, output_path: Path) -> None:
    """Plot approval rates by variant (OCR vs Image)."""
    plt.figure(figsize=(12, max(5, len(df) * 0.4)))

    # Sort by image approval rate
    df_sorted = df.sort_values("approval_rate_image", ascending=True)

    x = np.arange(len(df_sorted))
    width = 0.35

    plt.barh(x - width/2, df_sorted["approval_rate_ocr"], width, label="OCR Baseline", color="steelblue")
    plt.barh(x + width/2, df_sorted["approval_rate_image"], width, label="Image Variant", color="darkorange")

    plt.yticks(x, df_sorted["variant_id"])
    plt.xlabel("Approval Rate (% YES)")
    plt.title("Approval Rates by Typography Variant")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_flip_directionality_heatmap(paired: pd.DataFrame, output_path: Path) -> None:
    """Plot heatmap of flip directions by variant."""
    # Filter to only flips
    flipped = paired[paired["flip"] == 1].copy()

    if len(flipped) == 0:
        # No flips to plot
        return

    # Create pivot: variants x flip_direction
    pivot = flipped.pivot_table(
        index="variant_id",
        columns="flip_direction",
        values="flip",
        aggfunc="size",
        fill_value=0,
    )

    # Reorder columns if both directions present
    desired_order = ["NO→YES", "YES→NO"]
    pivot = pivot[[col for col in desired_order if col in pivot.columns]]

    plt.figure(figsize=(8, max(6, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="RdYlGn", cbar=True, linewidths=0.5)
    plt.title("Flip Directionality by Variant")
    plt.xlabel("Flip Direction")
    plt.ylabel("Variant")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_approval_rate_by_dimension(df: pd.DataFrame, output_path: Path) -> None:
    """Plot approval rates by dimension (OCR vs Image)."""
    plt.figure(figsize=(12, 5))

    # Sort by image approval rate
    df_sorted = df.sort_values("approval_rate_image", ascending=False)

    x = np.arange(len(df_sorted))
    width = 0.35

    plt.bar(x - width/2, df_sorted["approval_rate_ocr"], width, label="OCR Baseline", color="steelblue")
    plt.bar(x + width/2, df_sorted["approval_rate_image"], width, label="Image Variant", color="darkorange")

    plt.xticks(x, df_sorted["dimension_id"], rotation=45, ha="right")
    plt.ylabel("Approval Rate (% YES)")
    plt.title("Approval Rates by Dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def analyze_run(
    config: TypoEvalConfig,
    run_id: str,
    run_dir: Path,
    sentences_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Run full analysis on inference results.

    Writes CSVs, PNGs, and summary.md to analysis directory.
    """
    jsonl_path = run_dir / "raw" / "responses.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Results file not found: {jsonl_path}")

    results = load_results(jsonl_path)
    logger.info(f"Loaded {len(results)} results from {jsonl_path}")

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = analysis_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Build paired comparison table
    paired = build_delta_table(results, sentences_df)

    # Compute flip rates
    flip_rates = compute_flip_rates(paired)

    # Save flip rate CSVs
    for name, df in flip_rates.items():
        df.to_csv(analysis_dir / f"flip_{name}.csv", index=False)

    # Compute approval rates
    approval_rates = compute_approval_rates(paired)

    # Save approval rate CSVs
    for name, df in approval_rates.items():
        df.to_csv(analysis_dir / f"approval_rate_{name}.csv", index=False)

    # Compute flip directionality
    directionality = compute_flip_directionality(paired)

    # Save directionality CSVs
    for name, df in directionality.items():
        df.to_csv(analysis_dir / f"bias_direction_{name}.csv", index=False)

    # Bootstrap CIs for overall and by variant (flip rates)
    bootstrap_cfg = config.analysis.bootstrap
    overall_ci = bootstrap_ci(
        paired,
        column="flip",
        n_boot=bootstrap_cfg.n_boot,
        alpha=bootstrap_cfg.alpha,
    )
    overall_ci.to_csv(analysis_dir / "bootstrap_ci.csv", index=False)

    if "variant_id" in paired.columns:
        variant_ci = bootstrap_ci(
            paired,
            column="flip",
            group_col="variant_id",
            n_boot=bootstrap_cfg.n_boot,
            alpha=bootstrap_cfg.alpha,
        )
        variant_ci.to_csv(analysis_dir / "bootstrap_ci_by_variant.csv", index=False)

    # Bootstrap CIs for approval rates
    approval_ci = bootstrap_ci(
        paired,
        column="approval_image",
        n_boot=bootstrap_cfg.n_boot,
        alpha=bootstrap_cfg.alpha,
    )
    approval_ci.to_csv(analysis_dir / "bootstrap_ci_approval.csv", index=False)

    if "variant_id" in paired.columns:
        approval_variant_ci = bootstrap_ci(
            paired,
            column="approval_image",
            group_col="variant_id",
            n_boot=bootstrap_cfg.n_boot,
            alpha=bootstrap_cfg.alpha,
        )
        approval_variant_ci.to_csv(analysis_dir / "bootstrap_ci_approval_by_variant.csv", index=False)

    if "dimension_id" in paired.columns:
        approval_dimension_ci = bootstrap_ci(
            paired,
            column="approval_image",
            group_col="dimension_id",
            n_boot=bootstrap_cfg.n_boot,
            alpha=bootstrap_cfg.alpha,
        )
        approval_dimension_ci.to_csv(analysis_dir / "bootstrap_ci_approval_by_dimension.csv", index=False)

    # Generate figures
    if config.analysis.outputs.get("save_png", True):
        if "by_variant" in flip_rates:
            plot_flip_rate_by_variant(
                flip_rates["by_variant"],
                figures_dir / "flip_rate_by_variant.png",
            )

        if config.analysis.compute_heatmaps:
            plot_heatmap_sentence_variant(
                paired,
                figures_dir / "heatmap_sentence_variant.png",
            )

        # Approval rate plots
        if "by_variant" in approval_rates:
            plot_approval_rate_by_variant(
                approval_rates["by_variant"],
                figures_dir / "approval_rate_by_variant.png",
            )

        if "by_dimension" in approval_rates:
            plot_approval_rate_by_dimension(
                approval_rates["by_dimension"],
                figures_dir / "approval_rate_by_dimension.png",
            )

        # Flip directionality plot
        plot_flip_directionality_heatmap(
            paired,
            figures_dir / "bias_direction_heatmap.png",
        )

    # Generate summary markdown
    if config.analysis.outputs.get("save_md", True):
        summary = generate_summary_md(
            run_id=run_id,
            config=config,
            results=results,
            paired=paired,
            flip_rates=flip_rates,
            approval_rates=approval_rates,
            directionality=directionality,
            overall_ci=overall_ci,
            approval_ci=approval_ci,
        )
        (analysis_dir / "summary.md").write_text(summary)

    logger.info(f"Analysis written to {analysis_dir}")
    return analysis_dir


def generate_summary_md(
    run_id: str,
    config: TypoEvalConfig,
    results: pd.DataFrame,
    paired: pd.DataFrame,
    flip_rates: Dict[str, pd.DataFrame],
    approval_rates: Dict[str, pd.DataFrame],
    directionality: Dict[str, pd.DataFrame],
    overall_ci: pd.DataFrame,
    approval_ci: pd.DataFrame,
) -> str:
    """Generate summary markdown report."""
    lines = [
        "# Analysis Summary",
        "",
        f"## Run Metadata",
        "",
        f"- **Run ID**: {run_id}",
        f"- **Total responses**: {len(results)}",
        f"- **Paired comparisons**: {len(paired)}",
        f"- **Inference mode**: {config.inference.mode}",
        f"- **Temperature**: {config.inference.temperature}",
        "",
        "## Headline Results",
        "",
    ]

    # Overall flip rate with CI
    if len(overall_ci) > 0:
        row = overall_ci.iloc[0]
        lines.append(
            f"- **Overall flip rate**: {row['mean']:.3f} "
            f"(95% CI: [{row['ci_low']:.3f}, {row['ci_high']:.3f}], n={row['n']})"
        )
        lines.append("")

    # Overall approval rate with CI
    if len(approval_ci) > 0:
        row = approval_ci.iloc[0]
        lines.append(
            f"- **Overall approval rate (image)**: {row['mean']:.3f} "
            f"(95% CI: [{row['ci_low']:.3f}, {row['ci_high']:.3f}], n={row['n']})"
        )
        lines.append("")

    # Overall approval rates (OCR vs Image)
    if "overall" in approval_rates:
        row = approval_rates["overall"].iloc[0]
        lines.append(f"- **Approval rate (OCR baseline)**: {row['approval_rate_ocr']:.3f}")
        lines.append(f"- **Approval rate (Image variants)**: {row['approval_rate_image']:.3f}")
        lines.append("")

    # Flip directionality summary
    if "overall" in directionality and len(directionality["overall"]) > 0:
        lines.append("### Flip Directionality")
        lines.append("")
        for _, row in directionality["overall"].iterrows():
            lines.append(f"- **{row['flip_direction']}**: {row['count']} flips ({row['rate']:.1%})")
        lines.append("")

    # Top variants by flip rate
    if "by_variant" in flip_rates:
        lines.append("## Top Variants by Flip Rate")
        lines.append("")
        for _, row in flip_rates["by_variant"].head(10).iterrows():
            lines.append(f"- **{row['variant_id']}**: {row['flip_rate']:.3f}")
        lines.append("")

    # Top variants by approval rate
    if "by_variant" in approval_rates:
        lines.append("## Top Variants by Approval Rate (Image)")
        lines.append("")
        for _, row in approval_rates["by_variant"].head(10).iterrows():
            lines.append(
                f"- **{row['variant_id']}**: {row['approval_rate_image']:.3f} "
                f"(OCR: {row['approval_rate_ocr']:.3f})"
            )
        lines.append("")

    # Top boundary sentences
    if "by_sentence" in flip_rates:
        lines.append("## Top Boundary Sentences")
        lines.append("")
        for _, row in flip_rates["by_sentence"].head(10).iterrows():
            lines.append(f"- Sentence {int(row['sentence_id'])}: {row['flip_rate']:.3f}")
        lines.append("")

    # Figure links
    lines.extend([
        "## Figures",
        "",
        "### Flip Rates",
        "- [Flip Rate by Variant](figures/flip_rate_by_variant.png)",
        "- [Heatmap: Sentence x Variant](figures/heatmap_sentence_variant.png)",
        "",
        "### Approval Rates",
        "- [Approval Rate by Variant](figures/approval_rate_by_variant.png)",
        "- [Approval Rate by Dimension](figures/approval_rate_by_dimension.png)",
        "",
        "### Flip Directionality",
        "- [Directionality Heatmap](figures/bias_direction_heatmap.png)",
        "",
    ])

    return "\n".join(lines)
