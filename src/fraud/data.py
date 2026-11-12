"""Dataset loading with integrity checks."""

import logging
from pathlib import Path

import pandas as pd

from .config import (
    DATA_PATH,
    EXPECTED_COLS,
    EXPECTED_FRAUDS,
    EXPECTED_ROWS,
    FIGURES_DIR,
    RESULTS_DIR,
)

log = logging.getLogger(__name__)

# Legacy aliases for notebooks
RES = RESULTS_DIR
FIG = FIGURES_DIR


def load(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load creditcard.csv, verify shape and fraud count, sort by Time.

    Args:
        path: Path to the CSV file. Defaults to config.DATA_PATH.

    Returns:
        DataFrame sorted by Time with index reset.

    Raises:
        AssertionError: if shape, fraud count, or null checks fail.
    """
    df = pd.read_csv(path)
    assert df.shape == (EXPECTED_ROWS, EXPECTED_COLS), f"unexpected shape {df.shape}"
    assert df["Class"].sum() == EXPECTED_FRAUDS, f"unexpected fraud count {df['Class'].sum()}"
    assert df.isna().sum().sum() == 0, "nulls found"
    log.info("Loaded %d rows, %d frauds from %s", len(df), int(df["Class"].sum()), path)
    return df.sort_values("Time").reset_index(drop=True)
