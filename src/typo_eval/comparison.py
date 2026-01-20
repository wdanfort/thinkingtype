"""Cross-run comparison analysis for evaluating multiple model providers/configurations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typo_eval.analysis import load_results, build_delta_table, compute_flip_rates
from typo_eval.config import get_repo_root

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def find_runs(results_dir: Optional[Path] = None) -> List[Tuple[str, Path]]:
    """
    Find all run directories in the results folder.

    Returns list of (run_id, run_path) tuples.
    """
    if results_dir is None:
        results_dir = get_repo_root() / "results" / "runs"

    if not results_dir.exists():
        return []

    runs = []
    for run_path in results_dir.iterdir():
        if run_path.is_dir() and (run_path / "raw" / "responses.jsonl").exists():
            runs.append((run_path.name, run_path))

    return sorted(runs)


def load_run_metadata(run_path: Path) -> Dict:
    """
    Load metadata for a run from its JSONL file.

    Extracts provider, model, temperature, mode from first record.
    """
    jsonl_path = run_path / "raw" / "responses.jsonl"
    if not jsonl_path.exists():
        return {}

    with jsonl_path.open("r") as f:
        first_line = f.readline().strip()
        if first_line:
            try:
                record = json.loads(first_line)
                return {
                    "run_id": record.get("run_id", ""),
                    "provider": record.get("provider", "unknown"),
                    "model_text": record.get("model", "unknown"),
                    "temperature": record.get("temperature", 0.0),
                    "mode": record.get("mode", "unknown"),
                }
            except json.JSONDecodeError:
                pass

    return {}


def load_all_runs(
    run_ids: Optional[List[str]] = None,
    provider_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load and combine results from multiple runs.

    Args:
        run_ids: Specific run IDs to load. If None, load all runs.
        provider_filter: Filter to specific provider (e.g., "openai", "anthropic", "google")
        model_filter: Filter to specific model name substring
        results_dir: Custom results directory path

    Returns:
        DataFrame with all results combined, including run metadata columns.
    """
    available_runs = find_runs(results_dir)

    if not available_runs:
        logger.warning("No runs found")
        return pd.DataFrame()

    combined_results = []

    for run_id, run_path in available_runs:
        # Filter by run_ids if specified
        if run_ids and run_id not in run_ids:
            continue

        # Load metadata
        metadata = load_run_metadata(run_path)

        # Apply filters
        if provider_filter and metadata.get("provider") != provider_filter:
            continue
        if model_filter and model_filter not in metadata.get("model_text", ""):
            continue

        # Load results
        jsonl_path = run_path / "raw" / "responses.jsonl"
        try:
            results = load_results(jsonl_path)
            if len(results) > 0:
                logger.info(f"Loaded {len(results)} results from run {run_id}")
                combined_results.append(results)
        except Exception as exc:
            logger.warning(f"Failed to load run {run_id}: {exc}")
            continue

    if not combined_results:
        logger.warning("No results matched the filters")
        return pd.DataFrame()

    return pd.concat(combined_results, ignore_index=True)


def compare_flip_rates(
    run_ids: Optional[List[str]] = None,
    provider_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    sentences_df: Optional[pd.DataFrame] = None,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare flip rates across multiple runs.

    Returns DataFrame with columns:
        - run_id
        - provider
        - model
        - temperature
        - mode
        - overall_flip_rate
        - n_comparisons
    """
    all_results = load_all_runs(run_ids, provider_filter, model_filter, results_dir)

    if len(all_results) == 0:
        return pd.DataFrame()

    # Group by run_id and compute metrics
    comparison_rows = []

    for run_id in all_results["run_id"].unique():
        run_results = all_results[all_results["run_id"] == run_id]

        # Build paired comparison table
        try:
            paired = build_delta_table(run_results, sentences_df)

            # Compute overall flip rate
            overall_flip_rate = paired["flip"].mean() if len(paired) > 0 else np.nan
            n_comparisons = len(paired)

            # Extract metadata from first record
            first_record = run_results.iloc[0]

            comparison_rows.append({
                "run_id": run_id,
                "provider": first_record.get("provider", "unknown"),
                "model": first_record.get("model", "unknown"),
                "temperature": first_record.get("temperature", 0.0),
                "mode": first_record.get("mode", "unknown"),
                "overall_flip_rate": overall_flip_rate,
                "n_comparisons": n_comparisons,
            })
        except Exception as exc:
            logger.warning(f"Failed to analyze run {run_id}: {exc}")
            continue

    return pd.DataFrame(comparison_rows)


def compare_by_variant(
    run_ids: Optional[List[str]] = None,
    provider_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    sentences_df: Optional[pd.DataFrame] = None,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare flip rates by variant across multiple runs.

    Returns DataFrame with columns:
        - run_id
        - provider
        - model
        - variant_id
        - flip_rate
        - n
    """
    all_results = load_all_runs(run_ids, provider_filter, model_filter, results_dir)

    if len(all_results) == 0:
        return pd.DataFrame()

    comparison_rows = []

    for run_id in all_results["run_id"].unique():
        run_results = all_results[all_results["run_id"] == run_id]

        try:
            paired = build_delta_table(run_results, sentences_df)
            flip_rates = compute_flip_rates(paired)

            if "by_variant" in flip_rates:
                by_variant = flip_rates["by_variant"]

                # Extract metadata
                first_record = run_results.iloc[0]
                provider = first_record.get("provider", "unknown")
                model = first_record.get("model", "unknown")

                for _, row in by_variant.iterrows():
                    comparison_rows.append({
                        "run_id": run_id,
                        "provider": provider,
                        "model": model,
                        "variant_id": row["variant_id"],
                        "flip_rate": row["flip_rate"],
                        "n": row["n"],
                    })
        except Exception as exc:
            logger.warning(f"Failed to analyze run {run_id}: {exc}")
            continue

    return pd.DataFrame(comparison_rows)


def compare_by_dimension(
    run_ids: Optional[List[str]] = None,
    provider_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    sentences_df: Optional[pd.DataFrame] = None,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare flip rates by dimension across multiple runs.

    Returns DataFrame with columns:
        - run_id
        - provider
        - model
        - dimension_id
        - flip_rate
        - n
    """
    all_results = load_all_runs(run_ids, provider_filter, model_filter, results_dir)

    if len(all_results) == 0:
        return pd.DataFrame()

    comparison_rows = []

    for run_id in all_results["run_id"].unique():
        run_results = all_results[all_results["run_id"] == run_id]

        try:
            paired = build_delta_table(run_results, sentences_df)
            flip_rates = compute_flip_rates(paired)

            if "by_dimension" in flip_rates:
                by_dimension = flip_rates["by_dimension"]

                # Extract metadata
                first_record = run_results.iloc[0]
                provider = first_record.get("provider", "unknown")
                model = first_record.get("model", "unknown")

                for _, row in by_dimension.iterrows():
                    comparison_rows.append({
                        "run_id": run_id,
                        "provider": provider,
                        "model": model,
                        "dimension_id": row["dimension_id"],
                        "flip_rate": row["flip_rate"],
                        "n": row["n"],
                    })
        except Exception as exc:
            logger.warning(f"Failed to analyze run {run_id}: {exc}")
            continue

    return pd.DataFrame(comparison_rows)


def compare_approval_rates(
    run_ids: Optional[List[str]] = None,
    provider_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    sentences_df: Optional[pd.DataFrame] = None,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare approval rates across multiple runs.

    Returns DataFrame with columns:
        - run_id
        - provider
        - model
        - variant_id
        - approval_rate_text
        - approval_rate_image
        - n
    """
    from typo_eval.analysis import compute_approval_rates

    all_results = load_all_runs(run_ids, provider_filter, model_filter, results_dir)

    if len(all_results) == 0:
        return pd.DataFrame()

    comparison_rows = []

    for run_id in all_results["run_id"].unique():
        run_results = all_results[all_results["run_id"] == run_id]

        try:
            paired = build_delta_table(run_results, sentences_df)
            approval_rates = compute_approval_rates(paired)

            if "by_variant" in approval_rates:
                by_variant = approval_rates["by_variant"]

                # Extract metadata
                first_record = run_results.iloc[0]
                provider = first_record.get("provider", "unknown")
                model = first_record.get("model", "unknown")

                for _, row in by_variant.iterrows():
                    comparison_rows.append({
                        "run_id": run_id,
                        "provider": provider,
                        "model": model,
                        "variant_id": row["variant_id"],
                        "approval_rate_text": row["approval_rate_text"],
                        "approval_rate_image": row["approval_rate_image"],
                        "n": row["n"],
                    })
        except Exception as exc:
            logger.warning(f"Failed to analyze run {run_id}: {exc}")
            continue

    return pd.DataFrame(comparison_rows)


def compare_flip_directionality(
    run_ids: Optional[List[str]] = None,
    provider_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    sentences_df: Optional[pd.DataFrame] = None,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare flip directionality across multiple runs.

    Returns DataFrame with columns:
        - run_id
        - provider
        - model
        - variant_id
        - flip_direction
        - count
        - rate
    """
    from typo_eval.analysis import compute_flip_directionality

    all_results = load_all_runs(run_ids, provider_filter, model_filter, results_dir)

    if len(all_results) == 0:
        return pd.DataFrame()

    comparison_rows = []

    for run_id in all_results["run_id"].unique():
        run_results = all_results[all_results["run_id"] == run_id]

        try:
            paired = build_delta_table(run_results, sentences_df)
            directionality = compute_flip_directionality(paired)

            if "by_variant" in directionality:
                by_variant = directionality["by_variant"]

                # Extract metadata
                first_record = run_results.iloc[0]
                provider = first_record.get("provider", "unknown")
                model = first_record.get("model", "unknown")

                for _, row in by_variant.iterrows():
                    comparison_rows.append({
                        "run_id": run_id,
                        "provider": provider,
                        "model": model,
                        "variant_id": row["variant_id"],
                        "flip_direction": row["flip_direction"],
                        "count": row["count"],
                        "rate": row["rate"],
                    })
        except Exception as exc:
            logger.warning(f"Failed to analyze run {run_id}: {exc}")
            continue

    return pd.DataFrame(comparison_rows)


def plot_provider_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path,
    title: str = "Overall Flip Rate by Provider",
) -> None:
    """Plot comparison of flip rates across providers."""
    if len(comparison_df) == 0:
        logger.warning("No data to plot")
        return

    plt.figure(figsize=(10, 6))

    # Group by provider and plot
    providers = comparison_df["provider"].unique()
    x = np.arange(len(providers))

    means = []
    for provider in providers:
        provider_data = comparison_df[comparison_df["provider"] == provider]
        means.append(provider_data["overall_flip_rate"].mean())

    plt.bar(x, means, color=["steelblue", "darkorange", "forestgreen"][:len(providers)])
    plt.xticks(x, providers)
    plt.ylabel("Flip Rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved comparison plot to {output_path}")


def plot_variant_comparison(
    variant_df: pd.DataFrame,
    output_path: Path,
    title: str = "Flip Rate by Variant Across Providers",
) -> None:
    """Plot comparison of flip rates by variant across providers."""
    if len(variant_df) == 0:
        logger.warning("No data to plot")
        return

    # Pivot table: variant_id x provider
    pivot = variant_df.pivot_table(
        index="variant_id",
        columns="provider",
        values="flip_rate",
        aggfunc="mean",
    )

    plt.figure(figsize=(12, max(6, len(pivot) * 0.4)))
    pivot.plot(kind="barh", ax=plt.gca())
    plt.xlabel("Flip Rate")
    plt.ylabel("Variant")
    plt.title(title)
    plt.legend(title="Provider")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved variant comparison plot to {output_path}")


def plot_dimension_comparison(
    dimension_df: pd.DataFrame,
    output_path: Path,
    title: str = "Flip Rate by Dimension Across Providers",
) -> None:
    """Plot comparison of flip rates by dimension across providers."""
    if len(dimension_df) == 0:
        logger.warning("No data to plot")
        return

    # Pivot table: dimension_id x provider
    pivot = dimension_df.pivot_table(
        index="dimension_id",
        columns="provider",
        values="flip_rate",
        aggfunc="mean",
    )

    plt.figure(figsize=(12, 6))
    pivot.plot(kind="bar", ax=plt.gca())
    plt.xlabel("Dimension")
    plt.ylabel("Flip Rate")
    plt.title(title)
    plt.legend(title="Provider")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved dimension comparison plot to {output_path}")


def plot_approval_rate_comparison(
    approval_df: pd.DataFrame,
    output_path: Path,
    title: str = "Approval Rates by Provider",
) -> None:
    """Plot comparison of approval rates across providers."""
    if len(approval_df) == 0:
        logger.warning("No data to plot")
        return

    # Aggregate by provider
    provider_agg = approval_df.groupby("provider").agg({
        "approval_rate_text": "mean",
        "approval_rate_image": "mean",
    }).reset_index()

    plt.figure(figsize=(10, 6))
    x = np.arange(len(provider_agg))
    width = 0.35

    plt.bar(x - width/2, provider_agg["approval_rate_text"], width, label="Text Baseline", color="steelblue")
    plt.bar(x + width/2, provider_agg["approval_rate_image"], width, label="Image Variant", color="darkorange")

    plt.xticks(x, provider_agg["provider"])
    plt.ylabel("Approval Rate (% YES)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved approval rate comparison plot to {output_path}")


def plot_directionality_comparison(
    directionality_df: pd.DataFrame,
    output_path: Path,
    title: str = "Flip Directionality by Provider",
) -> None:
    """Plot comparison of flip directionality across providers."""
    if len(directionality_df) == 0:
        logger.warning("No data to plot")
        return

    # Aggregate counts by provider and direction
    provider_agg = directionality_df.groupby(["provider", "flip_direction"])["count"].sum().reset_index()

    # Pivot for stacked bar chart
    pivot = provider_agg.pivot(
        index="provider",
        columns="flip_direction",
        values="count"
    ).fillna(0)

    # Reorder columns
    desired_order = ["NO→YES", "YES→NO"]
    pivot = pivot[[col for col in desired_order if col in pivot.columns]]

    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True, color=["green", "red"], ax=plt.gca())
    plt.xlabel("Provider")
    plt.ylabel("Flip Count")
    plt.title(title)
    plt.legend(title="Direction")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved directionality comparison plot to {output_path}")


def plot_multi_provider_quadrant_scatter(
    all_results: pd.DataFrame,
    output_path: Path,
    title: str = "Typography Influence: Dimension vs Decision (All Providers)",
) -> None:
    """
    Plot quadrant scatter with all providers in one chart.

    Each dot represents one typography variant × one dimension × one provider.
    Different colors for different providers.

    X-axis: Dimension directionality (ΔYES_rate for that dimension: image − text)
    Y-axis: Decision directionality (ΔYES_rate for decision: image − text)
    """
    if len(all_results) == 0:
        logger.warning("No data to plot")
        return

    scatter_data = []

    # Process each run
    for run_id in all_results["run_id"].unique():
        run_results = all_results[all_results["run_id"] == run_id]

        # Build paired data for this run
        try:
            paired = build_delta_table(run_results)
        except Exception as exc:
            logger.warning(f"Failed to build paired data for run {run_id}: {exc}")
            continue

        # Separate dimensions and decision data
        dims_paired = paired[paired["dimension_id"].notna()].copy()
        decision_paired = paired[paired["dimension_id"].isna()].copy()

        if len(dims_paired) == 0 or len(decision_paired) == 0:
            logger.warning(f"Insufficient data for run {run_id}")
            continue

        # Calculate dimension directionality
        dims_directionality = (
            dims_paired.groupby(["variant_id", "dimension_id"])
            .agg(dimension_delta=("delta", "mean"))
            .reset_index()
        )

        # Calculate decision directionality
        decision_directionality = (
            decision_paired.groupby("variant_id")
            .agg(decision_delta=("delta", "mean"))
            .reset_index()
        )

        # Merge and add metadata
        merged = dims_directionality.merge(
            decision_directionality[["variant_id", "decision_delta"]],
            on="variant_id",
            how="inner",
        )

        if len(merged) > 0:
            # Extract provider and model from run results
            first_record = run_results.iloc[0]
            provider = first_record.get("provider", "unknown")
            model = first_record.get("model", "unknown")

            merged["provider"] = provider
            merged["model"] = model
            merged["run_id"] = run_id

            scatter_data.append(merged)

    if len(scatter_data) == 0:
        logger.warning("No scatter data to plot")
        return

    scatter_df = pd.concat(scatter_data, ignore_index=True)

    # Create scatter plot
    plt.figure(figsize=(12, 9))

    # Color palette for providers
    providers = scatter_df["provider"].unique()
    palette = sns.color_palette("tab10", n_colors=len(providers))
    provider_colors = dict(zip(providers, palette))

    for provider in providers:
        provider_data = scatter_df[scatter_df["provider"] == provider]
        plt.scatter(
            provider_data["dimension_delta"],
            provider_data["decision_delta"],
            alpha=0.6,
            s=50,
            c=[provider_colors[provider]],
            label=provider,
        )

    # Add quadrant lines at 0
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Labels
    plt.xlabel("Dimension Directionality\n(Δ Approval Rate: Image − Text)", fontsize=11)
    plt.ylabel("Decision Directionality\n(Δ Approval Rate: Image − Text)", fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold")

    # Add legend
    plt.legend(title="Provider", loc="best", fontsize=10)

    # Add grid
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved multi-provider quadrant scatter plot to {output_path}")


def run_comparison_analysis(
    output_dir: Path,
    run_ids: Optional[List[str]] = None,
    provider_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    sentences_df: Optional[pd.DataFrame] = None,
    results_dir: Optional[Path] = None,
) -> None:
    """
    Run full cross-run comparison analysis.

    Generates comparison tables and plots, saves to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    logger.info("Running cross-run comparison analysis...")

    # Load all results for multi-provider visualizations
    all_results = load_all_runs(run_ids, provider_filter, model_filter, results_dir)

    # Overall comparison
    overall_comparison = compare_flip_rates(
        run_ids, provider_filter, model_filter, sentences_df, results_dir
    )
    if len(overall_comparison) > 0:
        overall_comparison.to_csv(output_dir / "comparison_overall.csv", index=False)
        logger.info(f"Saved overall comparison to {output_dir / 'comparison_overall.csv'}")

        # Plot overall comparison
        plot_provider_comparison(
            overall_comparison,
            figures_dir / "comparison_overall.png",
        )

    # By variant comparison
    variant_comparison = compare_by_variant(
        run_ids, provider_filter, model_filter, sentences_df, results_dir
    )
    if len(variant_comparison) > 0:
        variant_comparison.to_csv(output_dir / "comparison_by_variant.csv", index=False)
        logger.info(f"Saved variant comparison to {output_dir / 'comparison_by_variant.csv'}")

        # Plot variant comparison
        plot_variant_comparison(
            variant_comparison,
            figures_dir / "comparison_by_variant.png",
        )

    # By dimension comparison
    dimension_comparison = compare_by_dimension(
        run_ids, provider_filter, model_filter, sentences_df, results_dir
    )
    if len(dimension_comparison) > 0:
        dimension_comparison.to_csv(output_dir / "comparison_by_dimension.csv", index=False)
        logger.info(f"Saved dimension comparison to {output_dir / 'comparison_by_dimension.csv'}")

        # Plot dimension comparison
        plot_dimension_comparison(
            dimension_comparison,
            figures_dir / "comparison_by_dimension.png",
        )

    # Approval rate comparison
    approval_comparison = compare_approval_rates(
        run_ids, provider_filter, model_filter, sentences_df, results_dir
    )
    if len(approval_comparison) > 0:
        approval_comparison.to_csv(output_dir / "comparison_approval_rates.csv", index=False)
        logger.info(f"Saved approval rate comparison to {output_dir / 'comparison_approval_rates.csv'}")

        # Plot approval rate comparison
        plot_approval_rate_comparison(
            approval_comparison,
            figures_dir / "comparison_approval_rates.png",
        )

    # Flip directionality comparison
    directionality_comparison = compare_flip_directionality(
        run_ids, provider_filter, model_filter, sentences_df, results_dir
    )
    if len(directionality_comparison) > 0:
        directionality_comparison.to_csv(output_dir / "comparison_directionality.csv", index=False)
        logger.info(f"Saved directionality comparison to {output_dir / 'comparison_directionality.csv'}")

        # Plot directionality comparison
        plot_directionality_comparison(
            directionality_comparison,
            figures_dir / "comparison_directionality.png",
        )

    # Multi-provider quadrant scatter plot
    plot_multi_provider_quadrant_scatter(
        all_results,
        figures_dir / "quadrant_scatter_multi_provider.png",
    )

    # Generate summary report
    summary_lines = [
        "# Cross-Run Comparison Analysis",
        "",
        "## Runs Compared",
        "",
    ]

    if len(overall_comparison) > 0:
        summary_lines.append(f"Total runs: {len(overall_comparison)}")
        summary_lines.append("")

        # Add directionality stats to main table
        summary_lines.append("| Run ID | Provider | Model | Flip Rate | Approval (Image) | NO→YES | YES→NO | N |")
        summary_lines.append("|--------|----------|-------|-----------|------------------|--------|--------|---|")

        for _, row in overall_comparison.iterrows():
            run_id = row['run_id']

            # Get approval rate for this run
            approval_rate = "N/A"
            if len(approval_comparison) > 0:
                run_approval = approval_comparison[approval_comparison['run_id'] == run_id]
                if len(run_approval) > 0:
                    approval_rate = f"{run_approval['approval_rate_image'].mean():.3f}"

            # Get directionality for this run
            no_yes, yes_no = "N/A", "N/A"
            if len(directionality_comparison) > 0:
                run_dir = directionality_comparison[directionality_comparison['run_id'] == run_id]
                if len(run_dir) > 0:
                    no_yes_count = run_dir[run_dir['flip_direction'] == 'NO→YES']['count'].sum()
                    yes_no_count = run_dir[run_dir['flip_direction'] == 'YES→NO']['count'].sum()
                    total_flips = no_yes_count + yes_no_count
                    if total_flips > 0:
                        no_yes = f"{no_yes_count/total_flips:.1%}"
                        yes_no = f"{yes_no_count/total_flips:.1%}"

            summary_lines.append(
                f"| {run_id} | {row['provider']} | {row['model']} | "
                f"{row['overall_flip_rate']:.3f} | {approval_rate} | {no_yes} | {yes_no} | {row['n_comparisons']} |"
            )
        summary_lines.append("")

    summary_lines.extend([
        "## Figures",
        "",
        "### Flip Rates",
        "- [Overall Comparison](figures/comparison_overall.png)",
        "- [Variant Comparison](figures/comparison_by_variant.png)",
        "- [Dimension Comparison](figures/comparison_by_dimension.png)",
        "",
        "### Approval Rates",
        "- [Approval Rate Comparison](figures/comparison_approval_rates.png)",
        "",
        "### Flip Directionality",
        "- [Directionality Comparison](figures/comparison_directionality.png)",
        "",
    ])

    (output_dir / "comparison_summary.md").write_text("\n".join(summary_lines))
    logger.info(f"Saved comparison summary to {output_dir / 'comparison_summary.md'}")

    logger.info("Cross-run comparison analysis complete!")
