"""Feature definitions and preprocessing transformers."""

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

V: list[str] = [f"V{i}" for i in range(1, 29)]

# Baseline feature set (no engineering)
FEATS_BASE: list[str] = ["Time", "Amount"] + V

# Engineered feature set: add hour_of_day and log_amount
FEATS_ENG: list[str] = ["Time", "Amount", "hour_of_day", "log_amount"] + V

# Default feature set used by all scripts (set by ablation results)
FEATS: list[str] = FEATS_BASE


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns to a copy of df. Safe to call multiple times.

    Adds:
      - hour_of_day: (Time // 3600) % 24
      - log_amount: log1p(Amount)
    """
    out = df.copy()
    out["hour_of_day"] = (out["Time"] // 3600) % 24
    out["log_amount"] = np.log1p(out["Amount"])
    return out


def make_pre(feats: Sequence[str] | None = None) -> ColumnTransformer:
    """Return a ColumnTransformer that scales Time and Amount; V features pass through.

    If engineered features are included, also scales hour_of_day and log_amount.

    Args:
        feats: Feature names to determine which columns are scaled. Defaults to FEATS.

    Returns:
        Fitted or unfitted ColumnTransformer.
    """
    if feats is None:
        feats = FEATS
    scale_cols = [c for c in ["Time", "Amount", "hour_of_day", "log_amount"] if c in feats]
    return ColumnTransformer(
        [("s", RobustScaler(), scale_cols)],
        remainder="passthrough",
    )
