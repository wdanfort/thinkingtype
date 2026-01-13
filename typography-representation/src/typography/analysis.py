"""Analysis and plotting."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from typography.io import ensure_dir, read_json, write_json
from typography.metrics import coerce_to_binary

sns.set_theme(style="whitegrid")


def analyze_run(run_id: str, runs_root: str, artifacts_root: str, logger) -> Path:
    run_dir = Path(runs_root) / run_id
    results_path = run_dir / "results.csv"
    run_json = read_json(run_dir / "run.json")
    artifact_set_id = run_json["artifact_set_id"]
    artifacts_dir = Path(run_json.get("artifact_dir", Path(artifacts_root) / artifact_set_id))
    metadata = pd.read_csv(artifacts_dir / "metadata.csv")
    results = pd.read_csv(results_path)

    paired = _build_delta_table(results, metadata)

    analysis_dir = run_dir / "analysis"
    tables_dir = ensure_dir(analysis_dir / "tables")
    figures_dir = ensure_dir(analysis_dir / "figures")

    mean_delta_by_dimension = (
        paired.groupby("dimension")["delta"].mean().reset_index().sort_values("delta")
    )
    flip_rate_by_dimension = (
        paired.groupby("dimension")["flip"].mean().reset_index().sort_values("flip", ascending=False)
    )
    variant_flip_rate_mean_delta = (
        paired.groupby("variant_id")
        .agg(flip_rate=("flip", "mean"), mean_delta=("delta", "mean"))
        .reset_index()
        .sort_values("flip_rate", ascending=False)
    )
    category_dimension_mean_delta = (
        paired.groupby(["category", "dimension"])["delta"].mean().reset_index()
    )
    category_dimension_mean_abs_delta = (
        paired.groupby(["category", "dimension"])["abs_delta"].mean().reset_index()
    )
    category_dimension_flip_rate = (
        paired.groupby(["category", "dimension"])["flip"].mean().reset_index()
    )
    overall_sensitivity_by_category = (
        paired.groupby("category")["abs_delta"].mean().reset_index().sort_values("abs_delta", ascending=False)
    )

    mean_delta_by_dimension.to_csv(tables_dir / "mean_delta_by_dimension.csv", index=False)
    flip_rate_by_dimension.to_csv(tables_dir / "flip_rate_by_dimension.csv", index=False)
    variant_flip_rate_mean_delta.to_csv(tables_dir / "variant_flip_rate_mean_delta.csv", index=False)
    category_dimension_mean_delta.to_csv(tables_dir / "category_dimension_mean_delta.csv", index=False)
    category_dimension_mean_abs_delta.to_csv(
        tables_dir / "category_dimension_mean_abs_delta.csv", index=False
    )
    category_dimension_flip_rate.to_csv(tables_dir / "category_dimension_flip_rate.csv", index=False)
    overall_sensitivity_by_category.to_csv(
        tables_dir / "overall_sensitivity_by_category.csv", index=False
    )

    split_half = _split_half_reliability(paired)
    pearson, spearman = _correlations(split_half)
    reliability_path = tables_dir / "split_half_reliability.json"
    write_json(
        reliability_path,
        {
            "pearson": pearson,
            "spearman": spearman,
            "created_at": dt.datetime.utcnow().isoformat(),
        },
    )

    _plot_mean_delta_by_dimension(mean_delta_by_dimension, figures_dir / "mean_delta_by_dimension.png")
    _plot_category_dimension_heatmap(
        category_dimension_mean_delta,
        figures_dir / "category_dimension_mean_delta_heatmap.png",
    )
    _plot_flip_rate_by_variant(
        variant_flip_rate_mean_delta,
        figures_dir / "flip_rate_by_variant.png",
    )

    summary_path = analysis_dir / "summary.md"
    summary_path.write_text(
        "# Analysis Summary\n\n"
        f"- Run ID: {run_id}\n"
        f"- Artifact set: {artifact_set_id}\n"
        f"- Rows analyzed: {len(paired)}\n\n"
        "## Reliability\n\n"
        f"- Split-half Pearson: {pearson:.3f}\n"
        f"- Split-half Spearman: {spearman:.3f}\n"
    )

    _update_registry_analysis(run_dir, run_id, runs_root, logger)
    logger.info("Analysis outputs written to %s", analysis_dir)
    return analysis_dir


def compare_runs(run_id_a: str, run_id_b: str, runs_root: str, artifacts_root: str, logger) -> Path:
    run_dir_a = Path(runs_root) / run_id_a
    run_dir_b = Path(runs_root) / run_id_b

    run_json_a = read_json(run_dir_a / "run.json")
    run_json_b = read_json(run_dir_b / "run.json")

    results_a = pd.read_csv(run_dir_a / "results.csv")
    results_b = pd.read_csv(run_dir_b / "results.csv")

    artifacts_dir_a = Path(run_json_a.get("artifact_dir", Path(artifacts_root) / run_json_a["artifact_set_id"]))
    artifacts_dir_b = Path(run_json_b.get("artifact_dir", Path(artifacts_root) / run_json_b["artifact_set_id"]))
    metadata_a = pd.read_csv(artifacts_dir_a / "metadata.csv")
    metadata_b = pd.read_csv(artifacts_dir_b / "metadata.csv")

    delta_a = _build_delta_table(results_a, metadata_a)
    delta_b = _build_delta_table(results_b, metadata_b)

    compare_root = Path(runs_root) / f"compare_{run_id_a}__vs__{run_id_b}"
    tables_dir = ensure_dir(compare_root / "tables")
    figures_dir = ensure_dir(compare_root / "figures")

    mean_delta_a = delta_a.groupby("dimension")["delta"].mean().rename("delta_runA")
    mean_delta_b = delta_b.groupby("dimension")["delta"].mean().rename("delta_runB")
    mean_delta_compare = pd.concat([mean_delta_a, mean_delta_b], axis=1).dropna()
    mean_delta_compare["delta_diff"] = mean_delta_compare["delta_runB"] - mean_delta_compare["delta_runA"]
    mean_delta_compare.reset_index().to_csv(
        tables_dir / "mean_delta_by_dimension_compare.csv", index=False
    )

    pearson_dim, spearman_dim = _correlations(mean_delta_compare)
    pd.DataFrame(
        [{"pearson": pearson_dim, "spearman": spearman_dim}]
    ).to_csv(tables_dir / "correlation_mean_delta_by_dimension.csv", index=False)

    cat_dim_a = (
        delta_a.groupby(["category", "dimension"])["delta"].mean().rename("delta_runA")
    )
    cat_dim_b = (
        delta_b.groupby(["category", "dimension"])["delta"].mean().rename("delta_runB")
    )
    cat_dim_compare = pd.concat([cat_dim_a, cat_dim_b], axis=1).dropna()
    cat_dim_compare["delta_diff"] = cat_dim_compare["delta_runB"] - cat_dim_compare["delta_runA"]
    cat_dim_compare.reset_index().to_csv(
        tables_dir / "category_dimension_mean_delta_compare.csv", index=False
    )

    pearson_cat, spearman_cat = _correlations(cat_dim_compare)
    pd.DataFrame(
        [{"pearson": pearson_cat, "spearman": spearman_cat}]
    ).to_csv(tables_dir / "correlation_category_dimension_mean_delta.csv", index=False)

    sens_a = (
        delta_a.groupby("category")["abs_delta"].mean().rename("mean_abs_delta_runA")
    )
    sens_b = (
        delta_b.groupby("category")["abs_delta"].mean().rename("mean_abs_delta_runB")
    )
    sens_compare = pd.concat([sens_a, sens_b], axis=1).dropna()
    sens_compare["diff"] = sens_compare["mean_abs_delta_runB"] - sens_compare["mean_abs_delta_runA"]
    sens_compare.reset_index().to_csv(
        tables_dir / "overall_sensitivity_by_category_compare.csv", index=False
    )

    _plot_compare_mean_delta(mean_delta_compare, figures_dir / "mean_delta_by_dimension_runA_vs_runB.png")
    _plot_compare_category_sensitivity(
        sens_compare, figures_dir / "category_sensitivity_runA_vs_runB.png"
    )

    write_json(
        compare_root / "comparison.json",
        {
            "runA": run_id_a,
            "runB": run_id_b,
            "created_at": dt.datetime.utcnow().isoformat(),
        },
    )

    logger.info("Comparison outputs written to %s", compare_root)
    return compare_root


def _build_delta_table(results: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["response_01"] = results["response_01"].apply(coerce_to_binary)
    results = results.dropna(subset=["response_01"]).copy()
    results["response_01"] = results["response_01"].astype(int)

    text_rows = results[results["representation"] == "text"]
    image_rows = results[results["representation"] == "image"]

    text_base = (
        text_rows.groupby(["item_id", "dimension"])["response_01"].mean().reset_index()
    )
    text_base["response_01"] = text_base["response_01"].round().astype(int)
    text_base = text_base.rename(columns={"response_01": "text"})

    image_rows = (
        image_rows.groupby(["item_id", "variant_id", "dimension"])["response_01"]
        .mean()
        .reset_index()
    )
    image_rows["response_01"] = image_rows["response_01"].round().astype(int)
    image_rows = image_rows.rename(columns={"response_01": "image"})

    paired = image_rows.merge(text_base, on=["item_id", "dimension"], how="left")
    paired = paired.dropna(subset=["text"]).copy()
    paired["text"] = paired["text"].astype(int)

    meta_unique = metadata.drop_duplicates(subset=["item_id"])
    paired = paired.merge(meta_unique[["item_id", "category"]], on="item_id", how="left")

    paired["delta"] = paired["image"] - paired["text"]
    paired["abs_delta"] = paired["delta"].abs()
    paired["flip"] = (paired["image"] != paired["text"]).astype(int)

    return paired


def _split_half_reliability(paired: pd.DataFrame, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    items = np.array(sorted(paired["item_id"].unique()))
    rng.shuffle(items)
    half = len(items) // 2
    s1 = set(items[:half])
    s2 = set(items[half:])

    a = paired[paired["item_id"].isin(s1)].groupby("dimension")["delta"].mean()
    b = paired[paired["item_id"].isin(s2)].groupby("dimension")["delta"].mean()
    out = pd.concat([a.rename("delta_half1"), b.rename("delta_half2")], axis=1).dropna()
    return out


def _correlations(df: pd.DataFrame) -> Tuple[float, float]:
    if df.shape[1] < 2 or df.shape[0] < 2:
        return float("nan"), float("nan")
    col1, col2 = df.columns[:2]
    pearson = pearsonr(df[col1], df[col2]).statistic
    spearman = spearmanr(df[col1], df[col2]).statistic
    return pearson, spearman


def _plot_mean_delta_by_dimension(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 4.5))
    plt.bar(df["dimension"], df["delta"], color="steelblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Δ (image - text)")
    plt.title("Mean delta by dimension")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_category_dimension_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.pivot(index="category", columns="dimension", values="delta")
    plt.figure(figsize=(12, 5))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, linewidths=0.3)
    plt.title("Category × dimension mean delta")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_flip_rate_by_variant(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(df["variant_id"], df["flip_rate"], color="darkorange")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Flip rate")
    plt.title("Flip rate by variant")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_compare_mean_delta(df: pd.DataFrame, out_path: Path) -> None:
    df = df.reset_index()
    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(10, 4.5))
    plt.bar(x - width / 2, df["delta_runA"], width, label="runA")
    plt.bar(x + width / 2, df["delta_runB"], width, label="runB")
    plt.xticks(x, df["dimension"], rotation=45, ha="right")
    plt.ylabel("Mean Δ")
    plt.title("Mean delta by dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_compare_category_sensitivity(df: pd.DataFrame, out_path: Path) -> None:
    df = df.reset_index()
    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width / 2, df["mean_abs_delta_runA"], width, label="runA")
    plt.bar(x + width / 2, df["mean_abs_delta_runB"], width, label="runB")
    plt.xticks(x, df["category"], rotation=30, ha="right")
    plt.ylabel("Mean |Δ|")
    plt.title("Category sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _update_registry_analysis(run_dir: Path, run_id: str, runs_root: str, logger) -> None:
    registry_path = Path(runs_root) / "index.csv"
    if not registry_path.exists():
        return
    registry = pd.read_csv(registry_path)
    if "analysis_dir" not in registry.columns:
        return
    registry.loc[registry["run_id"] == run_id, "analysis_dir"] = str(run_dir / "analysis")
    registry.to_csv(registry_path, index=False)
    logger.info("Updated registry with analysis path for %s", run_id)
