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
        summary_lines.append("| Run ID | Provider | Model | Flip Rate | N |")
        summary_lines.append("|--------|----------|-------|-----------|---|")
        for _, row in overall_comparison.iterrows():
            summary_lines.append(
                f"| {row['run_id']} | {row['provider']} | {row['model']} | "
                f"{row['overall_flip_rate']:.3f} | {row['n_comparisons']} |"
            )
        summary_lines.append("")

    summary_lines.extend([
        "## Figures",
        "",
        "- [Overall Comparison](figures/comparison_overall.png)",
        "- [Variant Comparison](figures/comparison_by_variant.png)",
        "- [Dimension Comparison](figures/comparison_by_dimension.png)",
        "",
    ])

    (output_dir / "comparison_summary.md").write_text("\n".join(summary_lines))
    logger.info(f"Saved comparison summary to {output_dir / 'comparison_summary.md'}")

    logger.info("Cross-run comparison analysis complete!")
