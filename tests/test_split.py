import numpy as np
import pandas as pd

from fraud.features import FEATS_BASE, make_pre
from fraud.split import cv_folds, time_split


def _make_df(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(0.0, index=range(n), columns=FEATS_BASE)
    df["Time"] = np.arange(n, dtype=float)
    df["Amount"] = rng.exponential(50, n)
    df["Class"] = 0
    return df


def test_no_time_overlap() -> None:
    tr, va, te = time_split(_make_df())
    assert tr.Time.max() < va.Time.min()
    assert va.Time.max() < te.Time.min()


def test_split_sizes() -> None:
    df = _make_df(1000)
    tr, va, te = time_split(df)
    assert len(tr) + len(va) + len(te) == 1000


def test_cv_folds_are_forward_only() -> None:
    for a, b in cv_folds(700):
        assert a.max() < b.min()


def test_scaler_fit_on_train_only() -> None:
    tr, va, _ = time_split(_make_df())
    pre = make_pre(FEATS_BASE).fit(tr[FEATS_BASE])
    center = pre.named_transformers_["s"].center_
    expected = tr[["Time", "Amount"]].median().values
    assert np.allclose(center, expected), f"{center} != {expected}"


def test_train_has_no_test_rows() -> None:
    """No index from test should appear in train."""
    tr, _, te = time_split(_make_df(500))
    assert len(set(tr.index) & set(te.index)) == 0
