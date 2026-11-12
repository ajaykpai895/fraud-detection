"""Cost calculation functions for the asymmetric fraud cost model.

Assumptions (overridable via parameters):
  - False negative cost = transaction Amount (money lost to fraud).
  - False positive cost = DEFAULT_FP_COST (analyst handling + customer friction).
  - True positive cost = DEFAULT_TP_COST (review is worth it; assumed 0).
"""

import logging

import numpy as np

from .config import DEFAULT_FP_COST, DEFAULT_TP_COST

log = logging.getLogger(__name__)


def total_cost(
    y: np.ndarray,
    pred: np.ndarray,
    amount: np.ndarray,
    fp_cost: float = DEFAULT_FP_COST,
    tp_cost: float = DEFAULT_TP_COST,
) -> float:
    """Compute total decision cost for a set of binary predictions.

    Args:
        y: true labels (1 = fraud).
        pred: predicted labels (1 = flagged for review).
        amount: transaction amounts (USD).
        fp_cost: cost per false positive alert (default $5.00).
        tp_cost: cost per true positive alert (default $0.00).

    Returns:
        Total cost in USD: sum of FN amounts + FP handling fees + TP costs.
    """
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    amount = np.asarray(amount, dtype=float)
    fn = y & ~pred
    fp = ~y & pred
    tp = y & pred
    return float(amount[fn].sum() + fp_cost * fp.sum() + tp_cost * tp.sum())


def flag_nothing_cost(y: np.ndarray, amount: np.ndarray) -> float:
    """Cost when no transactions are flagged (all frauds are missed).

    Args:
        y: true labels (1 = fraud).
        amount: transaction amounts (USD).

    Returns:
        Sum of amounts for all fraudulent transactions.
    """
    return float(np.asarray(amount, dtype=float)[np.asarray(y).astype(bool)].sum())


def best_threshold(
    y: np.ndarray,
    s: np.ndarray,
    amount: np.ndarray,
    fp_cost: float = DEFAULT_FP_COST,
    n: int = 500,
) -> tuple[float, float]:
    """Find the score threshold that minimises total_cost on the given set.

    Searches over n quantile-spaced thresholds. The threshold is chosen on
    the data passed in; callers must ensure this is the validation set, not
    the test set, to avoid leakage.

    Args:
        y: true labels.
        s: model scores.
        amount: transaction amounts (USD).
        fp_cost: false-positive handling cost.
        n: number of threshold candidates to evaluate.

    Returns:
        Tuple of (best_threshold, best_cost).
    """
    grid = np.unique(np.quantile(s, np.linspace(0.5, 0.9999, n)))
    grid = np.append(grid, s.max() + 1)  # lets 'flag nothing' win if cheapest
    costs = [total_cost(y, s >= t, amount, fp_cost) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])
