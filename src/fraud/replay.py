"""Replay the test period in time order to simulate production scoring.

Produces per-hour alert counts and cumulative cost curves.
"""

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DEFAULT_FP_COST, FIGURES_DIR, MODELS_DIR, RESULTS_DIR
from .data import load
from .features import FEATS_BASE
from .split import time_split

log = logging.getLogger(__name__)


def replay(
    te: pd.DataFrame,
    score: np.ndarray,
    thr: float,
    fp_cost: float = DEFAULT_FP_COST,
) -> pd.DataFrame:
    """Replay test transactions in time order, assigning costs per alert decision.

    Args:
        te: test-set DataFrame (must contain Time, Amount, Class columns).
        score: 1-D array of model scores, aligned with te rows.
        thr: decision threshold (flag if score >= thr).
        fp_cost: cost per false-positive alert (USD).

    Returns:
        DataFrame with columns: Time, Amount, Class, alert, hour,
        cost_model, cost_none, cum_model, cum_none.
    """
    r = te[["Time", "Amount", "Class"]].copy()
    r["alert"] = np.asarray(score) >= thr
    r["hour"] = ((r.Time - r.Time.min()) // 3600).astype(int)
    fraud = r.Class == 1
    r["cost_model"] = np.where(
        r.alert,
        np.where(fraud, 0.0, fp_cost),
        np.where(fraud, r.Amount, 0.0),
    )
    r["cost_none"] = np.where(fraud, r.Amount, 0.0)
    r["cum_model"] = r.cost_model.cumsum()
    r["cum_none"] = r.cost_none.cumsum()
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay the test period cost curve.")
    ap.add_argument("--threshold", type=float, help="Override decision threshold.")
    a = ap.parse_args()

    cfg_path = RESULTS_DIR / "chosen_threshold.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        fp_cost = cfg.get("fp_cost", DEFAULT_FP_COST)
        default_thr = cfg.get("threshold", 0.5)
    else:
        fp_cost = DEFAULT_FP_COST
        default_thr = 0.5

    thr = a.threshold if a.threshold is not None else default_thr

    model_path = MODELS_DIR / "xgb.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run scripts/run_modeling.py first."
        )
    model = joblib.load(model_path)
    _, _, te = time_split(load())

    score = model.predict_proba(te[FEATS_BASE])[:, 1]
    r = replay(te, score, thr, fp_cost)

    log.info("Alerts per hour:\n%s", r.groupby("hour")["alert"].sum().to_string())
    print(r.groupby("hour")["alert"].sum().rename("alerts_per_hour"))
    print(
        "final cost, model:",
        round(r.cum_model.iloc[-1], 2),
        " flag nothing:",
        round(r.cum_none.iloc[-1], 2),
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    r.plot(x="Time", y=["cum_model", "cum_none"], ax=ax)
    ax.set_xlabel("Seconds since first test transaction")
    ax.set_ylabel("Cumulative cost (USD)")
    ax.set_title("Cumulative cost: model vs flag-nothing (test period replay)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "replay_cost.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved replay_cost.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    main()
