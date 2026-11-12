import pandas as pd
import pytest

from fraud.config import EXPECTED_COLS, EXPECTED_FRAUDS, EXPECTED_ROWS
from fraud.data import load


def test_real_dataset_integrity():
    df = load()
    assert df.shape == (EXPECTED_ROWS, EXPECTED_COLS)
    assert df["Class"].sum() == EXPECTED_FRAUDS
    assert df.isna().sum().sum() == 0
    # Time must be monotonically non-decreasing after sort
    assert (df["Time"].diff().dropna() >= 0).all()


def test_load_fails_on_corrupt_data(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(bad_csv, index=False)
    with pytest.raises(AssertionError):
        load(bad_csv)
