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
from typo_eval.taxonomy import get_variant_attributes

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


def create_long_format_csv(
    results: pd.DataFrame,
    run_id: str,
    config: TypoEvalConfig,
    sentences_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Create long-format CSV with all run data in regularized columns.

    Columns:
    - run_id: Stable identifier for run
    - provider: openai / anthropic / google
    - model: Model identifier
    - temperature: Temperature setting
    - prompt_version: Hash/version of prompt templates
    - artifact_type: sentence | document
    - artifact_id: sentence_id or artifact_id
    - semantic_category: Semantic category (neutral, cta, etc.)
    - representation: text | image
    - variant_id: __text__ for text, variant ID for image
    - variant_bucket: font_family | emphasis | capitalization
    - font_family: Times | Arial | Comic | Monospace | OpenDyslexic
    - weight: regular | bold
    - caps: normal | all_caps
    - render_version: Hash/tag of render settings
    - task: decision | dimension
    - dimension: Dimension name (null for decision)
    - response: 0/1 parsed yes/no
    - raw_response: Raw model output
    - text_baseline: Baseline response from text (null for text rows)
    - flip_vs_text: 1 if flipped from text baseline (null for text rows)
    """
    rows = []

    # Get text baseline responses for flip calculation
    text_rows = results[results["representation"] == "text"].copy()

    # Build text baseline lookup: (artifact_id, dimension_id) -> response
    text_baseline_map = {}
    for _, row in text_rows.iterrows():
        artifact_id = row.get("sentence_id") or row.get("artifact_id")
        dimension_id = row.get("dimension_id")
        key = (artifact_id, dimension_id) if dimension_id else (artifact_id, None)
        text_baseline_map[key] = row.get("response_01", 0)

    # Process all rows
    for _, row in results.iterrows():
        artifact_id = row.get("sentence_id") or row.get("artifact_id")
        representation = row.get("representation", "unknown")
        variant_id = row.get("variant_id", "__text__")
        dimension_id = row.get("dimension_id")
        response = row.get("response_01", 0)

        # Get variant attributes from taxonomy
        variant_attrs = get_variant_attributes(variant_id)

        # Determine artifact type
        artifact_type = "sentence" if "sentence_id" in row else "document"

        # Get semantic category from sentences_df if available
        semantic_category = None
        if sentences_df is not None and artifact_type == "sentence":
            # sentences_df may have "id" or "sentence_id" as the key column
            id_col = "sentence_id" if "sentence_id" in sentences_df.columns else "id"
            matching = sentences_df[sentences_df[id_col] == artifact_id]
            if len(matching) > 0:
                semantic_category = matching.iloc[0].get("category")

        # Calculate text baseline and flip (only for image representation)
        text_baseline = None
        flip_vs_text = None
        if representation == "image":
            key = (artifact_id, dimension_id) if dimension_id else (artifact_id, None)
            text_baseline = text_baseline_map.get(key)
            if text_baseline is not None:
                flip_vs_text = 1 if response != text_baseline else 0

        # Determine task
        task = "dimension" if dimension_id else "decision"

        # Extract provider and model from row
        provider = row.get("provider", "unknown")
        model = row.get("model", "unknown")

        # Build row
        long_row = {
            "run_id": run_id,
            "provider": provider,
            "model": model,
            "temperature": config.inference.temperature,
            "prompt_version": "v1",  # TODO: compute hash from prompt templates
            "artifact_type": artifact_type,
            "artifact_id": str(artifact_id),
            "semantic_category": semantic_category,
            "representation": representation,
            "variant_id": variant_id,
            "variant_bucket": variant_attrs["variant_bucket"],
            "font_family": variant_attrs["font_family"],
            "weight": variant_attrs["weight"],
            "caps": variant_attrs["caps"],
            "render_version": "v1",  # TODO: compute hash from render config
            "task": task,
            "dimension": dimension_id,
            "response": int(response),
            "raw_response": row.get("response_raw", ""),
            "text_baseline": int(text_baseline) if text_baseline is not None else None,
            "flip_vs_text": int(flip_vs_text) if flip_vs_text is not None else None,
        }

        rows.append(long_row)

    return pd.DataFrame(rows)


def build_delta_table(
    results: pd.DataFrame,
    sentences_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build paired comparison table between text baseline and image variants.

    Returns DataFrame with columns:
    - sentence_id/artifact_id
    - variant_id
    - dimension_id (for dimensions mode)
    - text (baseline response)
    - image (image variant response)
    - delta (image - text)
    - flip (1 if different, 0 otherwise)
    - approval_text (YES rate for text baseline)
    - approval_image (YES rate for image variant)
    - flip_direction (NO→YES, YES→NO, or no_flip)
    """
    results = results.copy()

    # Coerce parsed_response to binary
    results["response_01"] = results["parsed_response"].apply(coerce_to_binary)
    results = results.dropna(subset=["response_01"]).copy()
    results["response_01"] = results["response_01"].astype(int)

    # Separate text and image results
    text_rows = results[results["representation"] == "text"]
    image_rows = results[results["representation"] == "image"]

    # Process dimensions mode and decision mode separately, then concatenate
    paired_parts = []

    # Process dimensions mode (has dimension_id)
    if "dimension_id" in results.columns:
        dims_text = text_rows[text_rows["dimension_id"].notna()]
        dims_image = image_rows[image_rows["dimension_id"].notna()]

        if len(dims_text) > 0 and len(dims_image) > 0:
            text_base = (
                dims_text.groupby(["sentence_id", "dimension_id"])["response_01"]
                .mean()
                .reset_index()
            )
            text_base["response_01"] = text_base["response_01"].round().astype(int)
            text_base = text_base.rename(columns={"response_01": "text"})

            # Get mode from first occurrence per sentence/dimension
            mode_map = dims_text.groupby(["sentence_id", "dimension_id"])["mode"].first().reset_index()
            text_base = text_base.merge(mode_map, on=["sentence_id", "dimension_id"], how="left")

            image_agg = (
                dims_image.groupby(["sentence_id", "variant_id", "dimension_id"])["response_01"]
                .mean()
                .reset_index()
            )
            image_agg["response_01"] = image_agg["response_01"].round().astype(int)
            image_agg = image_agg.rename(columns={"response_01": "image"})

            paired_dims = image_agg.merge(text_base, on=["sentence_id", "dimension_id"], how="left")
            paired_parts.append(paired_dims)

    # Process decision mode (no dimension_id or dimension_id is null)
    decision_text = text_rows[text_rows["dimension_id"].isna() if "dimension_id" in text_rows.columns else text_rows.index]
    decision_image = image_rows[image_rows["dimension_id"].isna() if "dimension_id" in image_rows.columns else image_rows.index]

    if len(decision_text) > 0 and len(decision_image) > 0:
        text_base = (
            decision_text.groupby(["sentence_id"])["response_01"]
            .mean()
            .reset_index()
        )
        text_base["response_01"] = text_base["response_01"].round().astype(int)
        text_base = text_base.rename(columns={"response_01": "text"})

        # Get mode from first occurrence per sentence
        mode_map = decision_text.groupby(["sentence_id"])["mode"].first().reset_index()
        text_base = text_base.merge(mode_map, on=["sentence_id"], how="left")

        image_agg = (
            decision_image.groupby(["sentence_id", "variant_id"])["response_01"]
            .mean()
            .reset_index()
        )
        image_agg["response_01"] = image_agg["response_01"].round().astype(int)
        image_agg = image_agg.rename(columns={"response_01": "image"})

        paired_decision = image_agg.merge(text_base, on=["sentence_id"], how="left")
        # Add dimension_id column as None for consistency
        if "dimension_id" not in paired_decision.columns:
            paired_decision["dimension_id"] = None
        paired_parts.append(paired_decision)

    # Concatenate all parts
    if len(paired_parts) == 0:
        raise ValueError("No valid paired comparisons found")
    paired = pd.concat(paired_parts, ignore_index=True)

    # Filter rows with missing text baseline
    paired = paired.dropna(subset=["text"]).copy()
    paired["text"] = paired["text"].astype(int)

    # Compute delta and flip
    paired["delta"] = paired["image"] - paired["text"]
    paired["abs_delta"] = paired["delta"].abs()
    paired["flip"] = (paired["image"] != paired["text"]).astype(int)

    # Add approval rate columns (YES = 1, NO = 0)
    paired["approval_text"] = paired["text"]
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

    # Add variant_bucket from taxonomy
    from typo_eval.taxonomy import get_variant_metadata
    paired["variant_bucket"] = paired["variant_id"].apply(
        lambda vid: get_variant_metadata(vid).variant_bucket
        if get_variant_metadata(vid)
        else "unknown"
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

    # By variant_bucket
    if "variant_bucket" in paired.columns:
        by_bucket = (
            paired.groupby("variant_bucket")
            .agg(flip_rate=("flip", "mean"), n=("flip", "size"))
            .reset_index()
            .sort_values("flip_rate", ascending=False)
        )
        results["by_variant_bucket"] = by_bucket

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

    Returns dict of DataFrames with approval_rate_text, approval_rate_image, and n.
    """
    results = {}

    # Overall approval rate
    overall = pd.DataFrame([{
        "approval_rate_text": paired["approval_text"].mean(),
        "approval_rate_image": paired["approval_image"].mean(),
        "n": len(paired),
    }])
    results["overall"] = overall

    # By variant
    by_variant = (
        paired.groupby("variant_id")
        .agg(
            approval_rate_text=("approval_text", "mean"),
            approval_rate_image=("approval_image", "mean"),
            n=("approval_text", "size")
        )
        .reset_index()
        .sort_values("approval_rate_image", ascending=False)
    )
    results["by_variant"] = by_variant

    # By variant_bucket
    if "variant_bucket" in paired.columns:
        by_bucket = (
            paired.groupby("variant_bucket")
            .agg(
                approval_rate_text=("approval_text", "mean"),
                approval_rate_image=("approval_image", "mean"),
                n=("approval_text", "size")
            )
            .reset_index()
            .sort_values("approval_rate_image", ascending=False)
        )
        results["by_variant_bucket"] = by_bucket

    # By sentence
    by_sentence = (
        paired.groupby("sentence_id")
        .agg(
            approval_rate_text=("approval_text", "mean"),
            approval_rate_image=("approval_image", "mean"),
            n=("approval_text", "size")
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
                approval_rate_text=("approval_text", "mean"),
                approval_rate_image=("approval_image", "mean"),
                n=("approval_text", "size")
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
                approval_rate_text=("approval_text", "mean"),
                approval_rate_image=("approval_image", "mean"),
                n=("approval_text", "size")
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

    plt.barh(x - width/2, df_sorted["approval_rate_text"], width, label="Text Baseline", color="steelblue")
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

    plt.bar(x - width/2, df_sorted["approval_rate_text"], width, label="Text Baseline", color="steelblue")
    plt.bar(x + width/2, df_sorted["approval_rate_image"], width, label="Image Variant", color="darkorange")

    plt.xticks(x, df_sorted["dimension_id"], rotation=45, ha="right")
    plt.ylabel("Approval Rate (% YES)")
    plt.title("Approval Rates by Dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_quadrant_scatter(
    paired: pd.DataFrame,
    output_path: Path,
    provider: str = "unknown",
    model: str = "unknown",
) -> None:
    """
    Plot quadrant scatter: Dimension directionality vs Decision directionality.

    Each dot represents one typography variant × one dimension.

    X-axis: Dimension directionality (ΔYES_rate for that dimension: image − text)
    Y-axis: Decision directionality (ΔYES_rate for decision: image − text)

    This reveals whether typography shifts that make a dimension more YES
    also make the decision more YES (upper-right quadrant = amplification).
    """
    # Separate dimensions and decision data
    dims_paired = paired[paired["dimension_id"].notna()].copy()
    decision_paired = paired[paired["dimension_id"].isna()].copy()

    if len(dims_paired) == 0 or len(decision_paired) == 0:
        logger.warning("Insufficient data for quadrant scatter plot")
        return

    # Calculate dimension directionality: Δ approval rate for each (variant, dimension)
    dims_directionality = (
        dims_paired.groupby(["variant_id", "dimension_id"])
        .agg(
            dimension_delta=("delta", "mean"),
            n=("delta", "size"),
        )
        .reset_index()
    )

    # Calculate decision directionality: Δ approval rate for each variant
    decision_directionality = (
        decision_paired.groupby("variant_id")
        .agg(
            decision_delta=("delta", "mean"),
            n=("delta", "size"),
        )
        .reset_index()
    )

    # Merge dimension and decision directionality
    scatter_data = dims_directionality.merge(
        decision_directionality[["variant_id", "decision_delta"]],
        on="variant_id",
        how="inner",
    )

    if len(scatter_data) == 0:
        logger.warning("No matching data for quadrant scatter plot")
        return

    # Create scatter plot
    plt.figure(figsize=(10, 8))

    # Use variant_bucket for coloring (from taxonomy)
    from typo_eval.taxonomy import get_variant_metadata

    scatter_data["variant_bucket"] = scatter_data["variant_id"].apply(
        lambda vid: get_variant_metadata(vid).variant_bucket
        if get_variant_metadata(vid)
        else "unknown"
    )

    # Color palette
    palette = {
        "font_family": "steelblue",
        "emphasis": "darkorange",
        "capitalization": "forestgreen",
        "unknown": "gray",
    }

    for bucket, group_df in scatter_data.groupby("variant_bucket"):
        color = palette.get(bucket, "gray")
        plt.scatter(
            group_df["dimension_delta"],
            group_df["decision_delta"],
            alpha=0.6,
            s=50,
            c=color,
            label=bucket,
        )

    # Add quadrant lines at 0
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Labels
    plt.xlabel("Dimension Directionality\n(Δ Approval Rate: Image − Text)", fontsize=11)
    plt.ylabel("Decision Directionality\n(Δ Approval Rate: Image − Text)", fontsize=11)
    plt.title(
        f"Typography Influence: Dimension vs Decision\n{provider} {model}",
        fontsize=12,
        fontweight="bold",
    )

    # Add legend
    plt.legend(title="Variant Bucket", loc="best")

    # Add grid
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved quadrant scatter plot to {output_path}")


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

    # Generate long-format CSV for flexible analysis
    long_csv = create_long_format_csv(results, run_id, config, sentences_df)
    long_csv.to_csv(analysis_dir / "results_long_format.csv", index=False)
    logger.info("Saved long-format CSV for flexible analysis")

    # Separate dimensions and decision mode data
    paired_dimensions = paired[paired["mode"] == "dimensions"].copy() if "mode" in paired.columns else paired.copy()
    paired_decision = paired[paired["mode"] == "decision"].copy() if "mode" in paired.columns else pd.DataFrame()

    # === DIMENSIONS MODE ANALYSIS ===
    # Compute flip rates
    flip_rates = compute_flip_rates(paired_dimensions)

    # Save flip rate CSVs
    for name, df in flip_rates.items():
        df.to_csv(analysis_dir / f"flip_{name}.csv", index=False)

    # Compute approval rates
    approval_rates = compute_approval_rates(paired_dimensions)

    # Save approval rate CSVs
    for name, df in approval_rates.items():
        df.to_csv(analysis_dir / f"approval_rate_{name}.csv", index=False)

    # Compute flip directionality
    directionality = compute_flip_directionality(paired_dimensions)

    # Save directionality CSVs
    for name, df in directionality.items():
        df.to_csv(analysis_dir / f"bias_direction_{name}.csv", index=False)

    # === DECISION MODE ANALYSIS ===
    decision_flip_rates = {}
    decision_approval_rates = {}
    decision_directionality = {}

    if len(paired_decision) > 0:
        # Compute decision mode metrics
        decision_flip_rates = compute_flip_rates(paired_decision)
        decision_approval_rates = compute_approval_rates(paired_decision)
        decision_directionality = compute_flip_directionality(paired_decision)

        # Save decision mode CSVs
        for name, df in decision_flip_rates.items():
            df.to_csv(analysis_dir / f"decision_flip_{name}.csv", index=False)

        for name, df in decision_approval_rates.items():
            df.to_csv(analysis_dir / f"decision_approval_rate_{name}.csv", index=False)

        for name, df in decision_directionality.items():
            df.to_csv(analysis_dir / f"decision_bias_direction_{name}.csv", index=False)

    # === BOOTSTRAP CIs FOR DIMENSIONS MODE ===
    bootstrap_cfg = config.analysis.bootstrap
    overall_ci = bootstrap_ci(
        paired_dimensions,
        column="flip",
        n_boot=bootstrap_cfg.n_boot,
        alpha=bootstrap_cfg.alpha,
    )
    overall_ci.to_csv(analysis_dir / "bootstrap_ci.csv", index=False)

    if "variant_id" in paired_dimensions.columns:
        variant_ci = bootstrap_ci(
            paired_dimensions,
            column="flip",
            group_col="variant_id",
            n_boot=bootstrap_cfg.n_boot,
            alpha=bootstrap_cfg.alpha,
        )
        variant_ci.to_csv(analysis_dir / "bootstrap_ci_by_variant.csv", index=False)

    # Bootstrap CIs for approval rates
    approval_ci = bootstrap_ci(
        paired_dimensions,
        column="approval_image",
        n_boot=bootstrap_cfg.n_boot,
        alpha=bootstrap_cfg.alpha,
    )
    approval_ci.to_csv(analysis_dir / "bootstrap_ci_approval.csv", index=False)

    if "variant_id" in paired_dimensions.columns:
        approval_variant_ci = bootstrap_ci(
            paired_dimensions,
            column="approval_image",
            group_col="variant_id",
            n_boot=bootstrap_cfg.n_boot,
            alpha=bootstrap_cfg.alpha,
        )
        approval_variant_ci.to_csv(analysis_dir / "bootstrap_ci_approval_by_variant.csv", index=False)

    if "dimension_id" in paired_dimensions.columns:
        approval_dimension_ci = bootstrap_ci(
            paired_dimensions,
            column="approval_image",
            group_col="dimension_id",
            n_boot=bootstrap_cfg.n_boot,
            alpha=bootstrap_cfg.alpha,
        )
        approval_dimension_ci.to_csv(analysis_dir / "bootstrap_ci_approval_by_dimension.csv", index=False)

    # === BOOTSTRAP CIs FOR DECISION MODE ===
    decision_overall_ci = pd.DataFrame()
    decision_variant_ci = pd.DataFrame()
    decision_approval_ci = pd.DataFrame()
    decision_approval_variant_ci = pd.DataFrame()

    if len(paired_decision) > 0:
        decision_overall_ci = bootstrap_ci(
            paired_decision,
            column="flip",
            n_boot=bootstrap_cfg.n_boot,
            alpha=bootstrap_cfg.alpha,
        )
        decision_overall_ci.to_csv(analysis_dir / "decision_bootstrap_ci.csv", index=False)

        if "variant_id" in paired_decision.columns:
            decision_variant_ci = bootstrap_ci(
                paired_decision,
                column="flip",
                group_col="variant_id",
                n_boot=bootstrap_cfg.n_boot,
                alpha=bootstrap_cfg.alpha,
            )
            decision_variant_ci.to_csv(analysis_dir / "decision_bootstrap_ci_by_variant.csv", index=False)

        # Decision approval CIs
        decision_approval_ci = bootstrap_ci(
            paired_decision,
            column="approval_image",
            n_boot=bootstrap_cfg.n_boot,
            alpha=bootstrap_cfg.alpha,
        )
        decision_approval_ci.to_csv(analysis_dir / "decision_bootstrap_ci_approval.csv", index=False)

        if "variant_id" in paired_decision.columns:
            decision_approval_variant_ci = bootstrap_ci(
                paired_decision,
                column="approval_image",
                group_col="variant_id",
                n_boot=bootstrap_cfg.n_boot,
                alpha=bootstrap_cfg.alpha,
            )
            decision_approval_variant_ci.to_csv(analysis_dir / "decision_bootstrap_ci_approval_by_variant.csv", index=False)

    # === GENERATE FIGURES ===
    if config.analysis.outputs.get("save_png", True):
        # Dimensions mode figures
        if "by_variant" in flip_rates:
            plot_flip_rate_by_variant(
                flip_rates["by_variant"],
                figures_dir / "flip_rate_by_variant.png",
            )

        if config.analysis.compute_heatmaps:
            plot_heatmap_sentence_variant(
                paired_dimensions,
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
            paired_dimensions,
            figures_dir / "bias_direction_heatmap.png",
        )

        # Quadrant scatter plot: Dimension vs Decision directionality
        if len(paired_decision) > 0:
            provider = results.iloc[0].get("provider", "unknown") if len(results) > 0 else "unknown"
            model = results.iloc[0].get("model", "unknown") if len(results) > 0 else "unknown"
            plot_quadrant_scatter(
                paired,  # Use full paired data (includes both dimensions and decision)
                figures_dir / "quadrant_scatter_dimension_vs_decision.png",
                provider=provider,
                model=model,
            )

        # Decision mode figures
        if len(paired_decision) > 0:
            if "by_variant" in decision_flip_rates:
                plot_flip_rate_by_variant(
                    decision_flip_rates["by_variant"],
                    figures_dir / "decision_flip_rate_by_variant.png",
                )

            if "by_variant" in decision_approval_rates:
                plot_approval_rate_by_variant(
                    decision_approval_rates["by_variant"],
                    figures_dir / "decision_approval_rate_by_variant.png",
                )

            plot_flip_directionality_heatmap(
                paired_decision,
                figures_dir / "decision_bias_direction_heatmap.png",
            )

    # Generate summary markdown
    if config.analysis.outputs.get("save_md", True):
        summary = generate_summary_md(
            run_id=run_id,
            config=config,
            results=results,
            paired_dimensions=paired_dimensions,
            paired_decision=paired_decision,
            flip_rates=flip_rates,
            approval_rates=approval_rates,
            directionality=directionality,
            overall_ci=overall_ci,
            approval_ci=approval_ci,
            decision_flip_rates=decision_flip_rates,
            decision_approval_rates=decision_approval_rates,
            decision_directionality=decision_directionality,
            decision_overall_ci=decision_overall_ci,
            decision_approval_ci=decision_approval_ci,
        )
        (analysis_dir / "summary.md").write_text(summary)

    logger.info(f"Analysis written to {analysis_dir}")
    return analysis_dir


def generate_summary_md(
    run_id: str,
    config: TypoEvalConfig,
    results: pd.DataFrame,
    paired_dimensions: pd.DataFrame,
    paired_decision: pd.DataFrame,
    flip_rates: Dict[str, pd.DataFrame],
    approval_rates: Dict[str, pd.DataFrame],
    directionality: Dict[str, pd.DataFrame],
    overall_ci: pd.DataFrame,
    approval_ci: pd.DataFrame,
    decision_flip_rates: Dict[str, pd.DataFrame],
    decision_approval_rates: Dict[str, pd.DataFrame],
    decision_directionality: Dict[str, pd.DataFrame],
    decision_overall_ci: pd.DataFrame,
    decision_approval_ci: pd.DataFrame,
) -> str:
    """Generate summary markdown report."""
    lines = [
        "# Analysis Summary",
        "",
        f"## Run Metadata",
        "",
        f"- **Run ID**: {run_id}",
        f"- **Total responses**: {len(results)}",
        f"- **Paired comparisons (dimensions)**: {len(paired_dimensions)}",
        f"- **Paired comparisons (decision)**: {len(paired_decision)}",
        f"- **Inference mode**: {config.inference.mode}",
        f"- **Temperature**: {config.inference.temperature}",
        "",
        "## Headline Results - Dimensions Mode",
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
        lines.append(f"- **Approval rate (OCR baseline)**: {row['approval_rate_text']:.3f}")
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
                f"(Text: {row['approval_rate_text']:.3f})"
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

    # === DECISION MODE SECTION ===
    if len(paired_decision) > 0:
        lines.extend([
            "---",
            "",
            "## Decision Mode Results",
            "",
            "Analysis of escalation decisions (binary: escalate vs. don't escalate).",
            "",
        ])

        # Decision flip rate with CI
        if len(decision_overall_ci) > 0:
            row = decision_overall_ci.iloc[0]
            lines.append(
                f"- **Decision flip rate**: {row['mean']:.3f} "
                f"(95% CI: [{row['ci_low']:.3f}, {row['ci_high']:.3f}], n={row['n']})"
            )
            lines.append("")

        # Decision approval rate with CI
        if len(decision_approval_ci) > 0:
            row = decision_approval_ci.iloc[0]
            lines.append(
                f"- **Decision approval rate (image)**: {row['mean']:.3f} "
                f"(95% CI: [{row['ci_low']:.3f}, {row['ci_high']:.3f}], n={row['n']})"
            )
            lines.append("")

        # Decision approval rates (OCR vs Image)
        if "overall" in decision_approval_rates:
            row = decision_approval_rates["overall"].iloc[0]
            lines.append(f"- **Decision approval rate (OCR baseline)**: {row['approval_rate_text']:.3f}")
            lines.append(f"- **Decision approval rate (Image variants)**: {row['approval_rate_image']:.3f}")
            lines.append("")

        # Decision directionality summary
        if "overall" in decision_directionality and len(decision_directionality["overall"]) > 0:
            lines.append("### Decision Flip Directionality")
            lines.append("")
            for _, row in decision_directionality["overall"].iterrows():
                lines.append(f"- **{row['flip_direction']}**: {row['count']} flips ({row['rate']:.1%})")
            lines.append("")

        # Top variants by decision flip rate
        if "by_variant" in decision_flip_rates:
            lines.append("### Top Variants by Decision Flip Rate")
            lines.append("")
            for _, row in decision_flip_rates["by_variant"].head(10).iterrows():
                lines.append(f"- **{row['variant_id']}**: {row['flip_rate']:.3f}")
            lines.append("")

        # Top variants by decision approval rate
        if "by_variant" in decision_approval_rates:
            lines.append("### Top Variants by Decision Approval Rate")
            lines.append("")
            for _, row in decision_approval_rates["by_variant"].head(10).iterrows():
                lines.append(
                    f"- **{row['variant_id']}**: {row['approval_rate_image']:.3f} "
                    f"(Text: {row['approval_rate_text']:.3f})"
                )
            lines.append("")

        # Decision mode figures
        lines.extend([
            "### Decision Mode Figures",
            "",
            "- [Decision Flip Rate by Variant](figures/decision_flip_rate_by_variant.png)",
            "- [Decision Approval Rate by Variant](figures/decision_approval_rate_by_variant.png)",
            "- [Decision Directionality Heatmap](figures/decision_bias_direction_heatmap.png)",
            "",
        ])

    return "\n".join(lines)
