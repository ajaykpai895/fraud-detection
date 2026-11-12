"""Rolling-origin (expanding window) temporal evaluation across sequential time blocks."""

import logging
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score

from .features import FEATS_BASE
from .models import iforest, logreg, xgb

log = logging.getLogger(__name__)


def rolling_origin_splits(
    df: pd.DataFrame,
    initial_frac: float = 0.60,
    step_frac: float = 0.10,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Generate rolling-origin (expanding train, fixed-size forward test) splits.

    Args:
        df: DataFrame in temporal order.
        initial_frac: Training fraction for period 1.
        step_frac: Test window fraction for each period.

    Returns:
        List of (period_label, train_df, test_df) tuples.
    """
    n = len(df)
    splits = []
    current_train_end = int(n * initial_frac)
    step_size = int(n * step_frac)

    split_idx = 1
    while current_train_end + step_size <= n:
        test_end = current_train_end + step_size
        train_df = df.iloc[:current_train_end]
        test_df = df.iloc[current_train_end:test_end]
        splits.append((f"period_{split_idx}", train_df, test_df))
        current_train_end = test_end
        split_idx += 1

    return splits


def run_rolling_evaluation(
    df: pd.DataFrame,
    xgb_params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and evaluate models across rolling temporal windows.

    Args:
        df: Full sorted dataset.
        xgb_params: Dictionary of XGBoost hyperparameters.

    Returns:
        Tuple of (results_by_period_df, summary_statistics_df).
    """
    splits = rolling_origin_splits(df, initial_frac=0.60, step_frac=0.10)

    if xgb_params is None:
        xgb_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 2,
            "reg_lambda": 5.0,
        }

    records = []
    for period_name, tr, te in splits:
        log.info("Evaluating rolling window: %s (train %d, test %d)", period_name, len(tr), len(te))
        Xtr, ytr = tr[FEATS_BASE], tr["Class"]
        Xte, yte = te[FEATS_BASE], te["Class"]
        no_skill = float(yte.mean())

        # 1. Logistic Regression
        lr = logreg(FEATS_BASE).fit(Xtr, ytr)
        s_lr = lr.predict_proba(Xte)[:, 1]
        ap_lr = float(average_precision_score(yte, s_lr))

        # 2. Isolation Forest
        ifo = iforest(n_estimators=100, max_samples=1024, max_features=1.0).fit(Xtr)
        s_if = -ifo.score_samples(Xte)
        ap_if = float(average_precision_score(yte, s_if))

        # 3. XGBoost
        model_xgb = xgb(xgb_params, ytr, feats=FEATS_BASE).fit(Xtr, ytr)
        s_xgb = model_xgb.predict_proba(Xte)[:, 1]
        ap_xgb = float(average_precision_score(yte, s_xgb))

        records.append(
            {
                "period": period_name,
                "train_rows": len(tr),
                "train_frauds": int(ytr.sum()),
                "test_rows": len(te),
                "test_frauds": int(yte.sum()),
                "no_skill_pr_auc": round(no_skill, 5),
                "logreg_pr_auc": round(ap_lr, 4),
                "iforest_pr_auc": round(ap_if, 4),
                "xgb_pr_auc": round(ap_xgb, 4),
            }
        )

    df_res = pd.DataFrame(records)

    summary_rows = []
    for model_col in ["no_skill_pr_auc", "logreg_pr_auc", "iforest_pr_auc", "xgb_pr_auc"]:
        vals = df_res[model_col]
        summary_rows.append(
            {
                "model": model_col.replace("_pr_auc", ""),
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
                "min": round(float(vals.min()), 4),
                "max": round(float(vals.max()), 4),
                "spread (max-min)": round(float(vals.max() - vals.min()), 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return df_res, summary_df
