"""Time-based train/validation/test splitting and cross-validation folds."""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import CV_SPLITS, TRAIN_FRAC, VAL_FRAC


def time_split(
    df: pd.DataFrame,
    train: float = TRAIN_FRAC,
    val: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into train / validation / test by row order (time-ordered).

    Args:
        df: DataFrame sorted by Time (assumed already sorted).
        train: fraction of rows for training.
        val: fraction of rows for validation.

    Returns:
        Tuple of (train_df, val_df, test_df). No row is in more than one split.
    """
    n = len(df)
    i = int(n * train)
    j = int(n * (train + val))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def cv_folds(n_train: int, n_splits: int = CV_SPLITS) -> list[tuple]:
    """Return expanding-window CV fold index pairs for the training set.

    Args:
        n_train: number of training rows.
        n_splits: number of folds (default from config).

    Returns:
        List of (train_indices, val_indices) tuples; val always follows train in time.
    """
    return list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_train)))
