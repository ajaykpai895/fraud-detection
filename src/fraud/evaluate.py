"""Evaluation metrics and bootstrap confidence interval estimators."""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .config import DEFAULT_FP_COST, DEFAULT_TP_COST, SEED
from .cost import total_cost

log = logging.getLogger(__name__)


def recall_at_precision(y: np.ndarray, s: np.ndarray, p: float = 0.80) -> float:
    """Compute maximum recall achievable at minimum precision p."""
    pr, rc, _ = precision_recall_curve(y, s)
    ok = rc[pr >= p]
    return float(ok.max()) if len(ok) else 0.0


def recall_at_fpr(y: np.ndarray, s: np.ndarray, f: float = 0.001) -> float:
    """Compute maximum true positive rate (recall) at maximum false positive rate f."""
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= f].max())


def summary(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    """Calculate core classification metrics on score s given true labels y."""
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "recall@precision80": float(recall_at_precision(y, s)),
        "recall@fpr0.1%": float(recall_at_fpr(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
    }


def bootstrap_pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, float]:
    """Compute point estimate and percentile bootstrap confidence intervals for PR-AUC."""
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    n = len(y)
    rng = np.random.default_rng(seed)

    boot_scores = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0 or y[idx].sum() == n:
            continue
        boot_scores.append(average_precision_score(y[idx], s[idx]))

    boot_scores_arr = np.asarray(boot_scores)
    point_est = float(average_precision_score(y, s))
    ci_lower = float(np.percentile(boot_scores_arr, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_scores_arr, 100 * (1 - alpha / 2)))

    return {
        "point": round(point_est, 4),
        "mean": round(float(np.mean(boot_scores_arr)), 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "std": round(float(np.std(boot_scores_arr)), 4),
    }


def bootstrap_cost(
    y_true: np.ndarray,
    pred: np.ndarray,
    amount: np.ndarray,
    fp_cost: float = DEFAULT_FP_COST,
    tp_cost: float = DEFAULT_TP_COST,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, float]:
    """Compute point estimate and percentile bootstrap confidence intervals for total cost."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(pred).astype(bool)
    amt = np.asarray(amount, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)

    point_cost = total_cost(y, p, amt, fp_cost=fp_cost, tp_cost=tp_cost)

    boot_costs = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        boot_costs.append(total_cost(y[idx], p[idx], amt[idx], fp_cost=fp_cost, tp_cost=tp_cost))

    boot_costs_arr = np.asarray(boot_costs)
    ci_lower = float(np.percentile(boot_costs_arr, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_costs_arr, 100 * (1 - alpha / 2)))

    return {
        "point": round(float(point_cost), 2),
        "mean": round(float(np.mean(boot_costs_arr)), 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "std": round(float(np.std(boot_costs_arr)), 2),
    }


def bootstrap_paired_diff(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, Any]:
    """Paired bootstrap hypothesis test for difference in PR-AUC (model A minus model B)."""
    y = np.asarray(y_true)
    sa = np.asarray(score_a)
    sb = np.asarray(score_b)
    n = len(y)
    rng = np.random.default_rng(seed)

    point_diff = float(average_precision_score(y, sa) - average_precision_score(y, sb))

    diffs = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0 or y[idx].sum() == n:
            continue
        ap_a = average_precision_score(y[idx], sa[idx])
        ap_b = average_precision_score(y[idx], sb[idx])
        diffs.append(ap_a - ap_b)

    diffs_arr = np.asarray(diffs)
    ci_lower = float(np.percentile(diffs_arr, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(diffs_arr, 100 * (1 - alpha / 2)))

    p_le = np.mean(diffs_arr <= 0)
    p_ge = np.mean(diffs_arr >= 0)
    p_val = float(min(1.0, 2.0 * min(p_le, p_ge)))

    return {
        "point_diff": round(point_diff, 4),
        "mean_diff": round(float(np.mean(diffs_arr)), 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_val, 4),
        "significant": bool(ci_lower > 0 or ci_upper < 0),
    }
