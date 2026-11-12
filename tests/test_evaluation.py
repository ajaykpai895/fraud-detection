import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fraud.evaluate import bootstrap_cost, bootstrap_paired_diff, bootstrap_pr_auc
from fraud.rolling import rolling_origin_splits


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    y = np.zeros(200, dtype=int)
    y[:20] = 1  # 10% positive class
    rng.shuffle(y)

    # score_a has overlap with negative class so PR-AUC is not trivially 1.0
    score_a = y * 0.4 + rng.uniform(0, 0.7, 200)
    # score_b is purely random noise
    score_b = rng.uniform(0, 1.0, 200)
    amounts = rng.exponential(100, 200)

    return y, score_a, score_b, amounts


def test_bootstrap_pr_auc_bounds(sample_data):
    y, score_a, _, _ = sample_data
    res = bootstrap_pr_auc(y, score_a, n_bootstraps=100, seed=42)

    assert res["ci_lower"] <= res["point"] <= res["ci_upper"]
    assert np.isclose(res["point"], average_precision_score(y, score_a), atol=1e-3)
    assert res["std"] > 0


def test_bootstrap_cost_bounds(sample_data):
    y, score_a, _, amounts = sample_data
    pred = score_a > 0.6
    res = bootstrap_cost(y, pred, amounts, fp_cost=5.0, n_bootstraps=100, seed=42)

    assert res["ci_lower"] <= res["point"] <= res["ci_upper"]
    assert res["std"] > 0


def test_bootstrap_paired_diff_identical_models(sample_data):
    y, score_a, _, _ = sample_data
    res = bootstrap_paired_diff(y, score_a, score_a, n_bootstraps=50, seed=42)

    assert res["point_diff"] == 0.0
    assert res["ci_lower"] <= 0.0 <= res["ci_upper"]
    assert not res["significant"]


def test_bootstrap_paired_diff_different_models(sample_data):
    y, score_a, score_b, _ = sample_data
    res = bootstrap_paired_diff(y, score_a, score_b, n_bootstraps=100, seed=42)

    assert res["point_diff"] > 0
    assert res["ci_lower"] > 0  # score_a should be significantly better than pure noise
    assert res["significant"]


def test_rolling_origin_splits_forward_only():
    df = pd.DataFrame(
        {
            "Time": np.arange(1000, dtype=float),
            "Amount": np.ones(1000),
            "Class": np.zeros(1000),
        }
    )
    splits = rolling_origin_splits(df, initial_frac=0.60, step_frac=0.10)

    assert len(splits) == 4
    prev_train_len = 0
    for name, tr, te in splits:
        assert len(tr) > prev_train_len, "Train must expand"
        assert tr.Time.max() < te.Time.min(), "Train must strictly precede test"
        prev_train_len = len(tr)
