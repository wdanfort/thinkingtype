from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from typography_eval import config
from typography_eval.src import io_utils
from typography_eval.src.metrics import parse_to_01


def load_results(run_id: Optional[str] = None, run_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    df = io_utils.load_csv(config.RESULTS_PATH)
    if df.empty:
        return df
    if run_id:
        return df[df["run_id"] == run_id].copy()
    if run_ids:
        return df[df["run_id"].isin(list(run_ids))].copy()
    return df


def _ensure_response_01(df: pd.DataFrame) -> pd.DataFrame:
    if "response_01" in df.columns:
        df["response_01"] = pd.to_numeric(df["response_01"], errors="coerce")
        return df
    resp_col = "response_norm" if "response_norm" in df.columns else "response_raw"
    df["response_01"] = df[resp_col].apply(lambda v: parse_to_01(v)[0])
    return df


def compute_delta(df_results: pd.DataFrame) -> pd.DataFrame:
    df = df_results.copy()
    df["representation"] = df["representation"].astype(str).str.lower().str.strip()
    df["dimension"] = df["dimension"].astype(str).str.strip()
    df["variant_id"] = df["variant_id"].astype(str)

    df = _ensure_response_01(df)
    df = df.dropna(subset=["response_01"]).copy()
    df["response_01"] = df["response_01"].astype(int)

    ocr = (
        df[df["representation"] == "ocr"]
        .groupby(["sentence_id", "dimension"])["response_01"]
        .mean()
        .reset_index()
        .rename(columns={"response_01": "ocr"})
    )
    ocr["ocr"] = ocr["ocr"].round().astype(int)

    img = (
        df[df["representation"] == "image"]
        .groupby(["sentence_id", "variant_id", "dimension"])["response_01"]
        .mean()
        .reset_index()
        .rename(columns={"response_01": "image"})
    )
    img["image"] = img["image"].round().astype(int)

    paired = img.merge(ocr, on=["sentence_id", "dimension"], how="left")
    paired = paired.dropna(subset=["ocr"]).copy()
    paired["ocr"] = paired["ocr"].astype(int)

    paired["delta"] = paired["image"] - paired["ocr"]
    paired["abs_delta"] = paired["delta"].abs()
    paired["flip"] = (paired["image"] != paired["ocr"]).astype(int)

    sentences = io_utils.load_sentences()
    paired = paired.merge(
        sentences[["sentence_id", "sentence_category"]], on="sentence_id", how="left"
    )

    variants = io_utils.load_variants()
    paired = paired.merge(
        variants[["variant_id", "variant_group"]], on="variant_id", how="left"
    )

    return paired


def summarize_by_dimension(df_delta: pd.DataFrame) -> pd.DataFrame:
    return (
        df_delta.groupby("dimension")
        .agg(
            mean_delta=("delta", "mean"),
            mean_abs_delta=("abs_delta", "mean"),
            flip_rate=("flip", "mean"),
            n=("delta", "size"),
        )
        .reset_index()
    )


def summarize_by_sentence_category(df_delta: pd.DataFrame) -> pd.DataFrame:
    return (
        df_delta.groupby(["sentence_category", "dimension"])
        .agg(
            mean_delta=("delta", "mean"),
            mean_abs_delta=("abs_delta", "mean"),
            flip_rate=("flip", "mean"),
            n=("delta", "size"),
        )
        .reset_index()
    )


def summarize_by_variant(df_delta: pd.DataFrame) -> pd.DataFrame:
    return (
        df_delta.groupby(["variant_id", "dimension"])
        .agg(
            mean_delta=("delta", "mean"),
            mean_abs_delta=("abs_delta", "mean"),
            flip_rate=("flip", "mean"),
            n=("delta", "size"),
        )
        .reset_index()
    )


def compare_runs(run_id_a: str, run_id_b: str) -> dict:
    df_a = compute_delta(load_results(run_id=run_id_a))
    df_b = compute_delta(load_results(run_id=run_id_b))

    dim_a = df_a.groupby("dimension")["delta"].mean().rename("delta_a")
    dim_b = df_b.groupby("dimension")["delta"].mean().rename("delta_b")
    dim_cmp = pd.concat([dim_a, dim_b], axis=1).dropna()

    cat_a = (
        df_a.groupby(["sentence_category", "dimension"])["delta"]
        .mean()
        .rename("delta_a")
    )
    cat_b = (
        df_b.groupby(["sentence_category", "dimension"])["delta"]
        .mean()
        .rename("delta_b")
    )
    cat_cmp = pd.concat([cat_a, cat_b], axis=1).dropna()

    return {
        "dimension": {
            "pearson": dim_cmp.corr(method="pearson").iloc[0, 1],
            "spearman": dim_cmp.corr(method="spearman").iloc[0, 1],
            "rows": dim_cmp.reset_index(),
        },
        "sentence_category_dimension": {
            "pearson": cat_cmp.corr(method="pearson").iloc[0, 1],
            "spearman": cat_cmp.corr(method="spearman").iloc[0, 1],
            "rows": cat_cmp.reset_index(),
        },
    }
