"""Statistical primitives: exact McNemar, BH-FDR, and cluster bootstrap.

The unit of independence in this experiment is the sentence: each sentence
contributes many paired comparisons (variants x dimensions), so per-pair
tests understate the variance. The cluster bootstrap resamples sentences
with replacement (including multiplicity) and is the primary inference
tool; exact McNemar on discordant pairs is reported as a secondary,
anti-conservative check.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def mcnemar_exact(n01: int, n10: int) -> float:
    """
    Exact McNemar test p-value from discordant pair counts.

    n01: pairs that moved NO->YES (text 0, image 1)
    n10: pairs that moved YES->NO (text 1, image 0)

    Under H0 (no systematic direction), discordant pairs split 50/50.
    Returns two-sided p-value; NaN if there are no discordant pairs.
    """
    n_discordant = n01 + n10
    if n_discordant == 0:
        return float("nan")
    result = scipy_stats.binomtest(min(n01, n10), n_discordant, 0.5, alternative="two-sided")
    return float(result.pvalue)


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    """
    Benjamini-Hochberg adjusted p-values (q-values).

    NaN inputs are passed through as NaN and excluded from the adjustment.
    """
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan)
    mask = ~np.isnan(pvals)
    if mask.sum() > 0:
        qvals[mask] = scipy_stats.false_discovery_control(pvals[mask], method="bh")
    return qvals


def per_cluster_counts(
    df: pd.DataFrame,
    cluster_col: str,
    count_cols: Sequence[str],
) -> np.ndarray:
    """
    Sum count columns within each cluster.

    Returns an (n_clusters, len(count_cols)) array suitable for
    cluster_bootstrap_ci.
    """
    grouped = df.groupby(cluster_col)[list(count_cols)].sum()
    return grouped.to_numpy(dtype=float)


def cluster_bootstrap_ci(
    cluster_counts: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for a statistic of summed cluster counts.

    cluster_counts: (n_clusters, k) array of per-cluster counts.
    stat_fn: maps a length-k vector of summed counts to a scalar
        (may return NaN for degenerate resamples, which are dropped).

    Clusters are resampled with replacement *including multiplicity*
    (a cluster drawn twice is counted twice), which is the correct
    cluster bootstrap.
    """
    n_clusters = len(cluster_counts)
    if n_clusters == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        val = stat_fn(cluster_counts[idx].sum(axis=0))
        if not np.isnan(val):
            values.append(val)

    if not values:
        return float("nan"), float("nan")

    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def rate_stat(sums: np.ndarray) -> float:
    """Rate = successes / total for count vector [successes, total]."""
    return sums[0] / sums[1] if sums[1] > 0 else float("nan")


def net_bias_stat(sums: np.ndarray) -> float:
    """Net direction in percent for count vector [no_to_yes, yes_to_no]."""
    total = sums[0] + sums[1]
    if total == 0:
        return float("nan")
    return 100.0 * (sums[0] - sums[1]) / total


def lift_stat(sums: np.ndarray) -> float:
    """
    Lift for count vector [a, b, c, d]:
    a = dim flip & decision flip, b = dim flip & no decision flip,
    c = no dim flip & decision flip, d = no dim flip & no decision flip.
    Lift = P(dec flip | dim flip) / P(dec flip | no dim flip).
    """
    if (sums[0] + sums[1]) == 0 or (sums[2] + sums[3]) == 0:
        return float("nan")
    p1 = sums[0] / (sums[0] + sums[1])
    p0 = sums[2] / (sums[2] + sums[3])
    if p0 == 0:
        return float("nan")
    return p1 / p0
