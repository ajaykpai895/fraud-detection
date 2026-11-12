"""Model constructors, training pipelines, and hyperparameter tuning."""

import logging
from typing import Any, Sequence

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import N_OPTUNA_TRIALS, SEED
from .features import make_pre

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def logreg(feats: Sequence[str] | None = None) -> Pipeline:
    """Logistic regression with class_weight='balanced' inside preprocessing pipeline."""
    return Pipeline(
        [
            ("pre", make_pre(feats)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)),
        ]
    )


def iforest(**kw: Any) -> IsolationForest:
    """Unsupervised Isolation Forest baseline."""
    return IsolationForest(random_state=SEED, n_jobs=-1, **kw)


def spw(y: pd.Series | np.ndarray) -> float:
    """Calculate scale_pos_weight = (negative count) / (positive count)."""
    y_arr = np.asarray(y)
    return float((y_arr == 0).sum() / max((y_arr == 1).sum(), 1))


def xgb(
    params: dict[str, Any],
    y: pd.Series | np.ndarray | None = None,
    weighted: bool = True,
    feats: Sequence[str] | None = None,
) -> Pipeline:
    """Construct XGBoost Classifier wrapped in a preprocessing pipeline."""
    kw = dict(
        params,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    if weighted and y is not None:
        kw["scale_pos_weight"] = spw(y)
    return Pipeline(
        [
            ("pre", make_pre(feats)),
            ("clf", XGBClassifier(**kw)),
        ]
    )


def xgb_smote(
    params: dict[str, Any],
    feats: Sequence[str] | None = None,
) -> ImbPipeline:
    """Construct XGBoost with SMOTE oversampling in an imblearn pipeline."""
    kw = dict(
        params,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    return ImbPipeline(
        [
            ("pre", make_pre(feats)),
            ("smote", SMOTE(random_state=SEED)),
            ("clf", XGBClassifier(**kw)),
        ]
    )


def random_forest(feats: Sequence[str] | None = None) -> Pipeline:
    """Random Forest with balanced sub-sampling inside preprocessing pipeline."""
    return Pipeline(
        [
            ("pre", make_pre(feats)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ]
    )


def lgbm(
    params: dict[str, Any] | None = None,
    y: pd.Series | np.ndarray | None = None,
    feats: Sequence[str] | None = None,
) -> Pipeline:
    """LightGBM Classifier wrapped in preprocessing pipeline."""
    base: dict[str, Any] = dict(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_lambda=5.0,
        n_jobs=-1,
        random_state=SEED,
        verbosity=-1,
    )
    if params:
        base.update(params)
    if y is not None:
        base["scale_pos_weight"] = spw(y)
    return Pipeline(
        [
            ("pre", make_pre(feats)),
            ("clf", LGBMClassifier(**base)),
        ]
    )


def rank_avg_ensemble(scores_list: Sequence[np.ndarray]) -> np.ndarray:
    """Rank-average ensemble: average normalized percentile ranks across score arrays.

    Args:
        scores_list: Sequence of 1-D numpy arrays of identical length.

    Returns:
        1-D numpy array of averaged rank scores in [0, 1].
    """
    n = len(scores_list[0])
    ranked = []
    for s in scores_list:
        order = np.argsort(s)
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        ranked.append(ranks / n)
    return np.mean(ranked, axis=0)


def tune_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple],
    n_trials: int = N_OPTUNA_TRIALS,
) -> optuna.Study:
    """Tune XGBoost hyperparameters with Optuna over time-series cross validation folds."""

    def objective(t: optuna.Trial) -> float:
        p = dict(
            n_estimators=t.suggest_int("n_estimators", 200, 800),
            max_depth=t.suggest_int("max_depth", 3, 8),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=t.suggest_int("min_child_weight", 1, 10),
            reg_lambda=t.suggest_float("reg_lambda", 1e-2, 10, log=True),
        )
        out = []
        for a, b in folds:
            m = xgb(p, y.iloc[a], feats=list(X.columns)).fit(X.iloc[a], y.iloc[a])
            out.append(average_precision_score(y.iloc[b], m.predict_proba(X.iloc[b])[:, 1]))
        score = float(np.mean(out))
        log.info("Optuna trial %d/%d: mean cv pr-auc = %.4f", t.number + 1, n_trials, score)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials)
    return study
