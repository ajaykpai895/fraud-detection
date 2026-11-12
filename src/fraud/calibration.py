"""Probability calibration utilities and capacity-constrained decision rules."""

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def fit_platt(y_val: np.ndarray, s_val: np.ndarray) -> LogisticRegression:
    """Fit Platt scaling (logistic regression on raw scores) on validation set."""
    s = np.asarray(s_val).reshape(-1, 1)
    y = np.asarray(y_val)
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(s, y)
    return lr


def fit_isotonic(y_val: np.ndarray, s_val: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression calibration on validation set."""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(np.asarray(s_val), np.asarray(y_val, dtype=float))
    return ir


def apply_calibrator(cal: Any, s: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator (Platt or Isotonic) to raw model scores."""
    s_arr = np.asarray(s)
    if isinstance(cal, LogisticRegression):
        return cal.predict_proba(s_arr.reshape(-1, 1))[:, 1]
    return cal.predict(s_arr)


def brier(y: np.ndarray, s: np.ndarray) -> float:
    """Compute Brier score loss (mean squared probability error)."""
    return float(brier_score_loss(np.asarray(y), np.asarray(s)))


def reliability_data(
    y: np.ndarray,
    s: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fraction of positives and mean predicted probabilities per bin."""
    frac_pos, mean_pred = calibration_curve(
        np.asarray(y), np.asarray(s), n_bins=n_bins, strategy="quantile"
    )
    return frac_pos, mean_pred


def amount_aware_threshold(
    p_cal: np.ndarray,
    amount: np.ndarray,
    fp_cost: float,
) -> np.ndarray:
    """Flag transaction if expected fraud savings (p_cal * Amount) exceeds false-alarm cost."""
    return np.asarray(p_cal) * np.asarray(amount, dtype=float) > fp_cost


def capacity_constrained_flags(
    s: np.ndarray,
    n_per_hour: int,
    hour_indices: np.ndarray,
) -> np.ndarray:
    """Flag top n_per_hour highest-risk transactions within each hourly period."""
    s_arr = np.asarray(s)
    h_arr = np.asarray(hour_indices)
    flags = np.zeros(len(s_arr), dtype=bool)
    for h in np.unique(h_arr):
        mask = h_arr == h
        idx = np.where(mask)[0]
        ranked = idx[np.argsort(-s_arr[idx])]
        top_n = ranked[:n_per_hour]
        flags[top_n] = True
    return flags
