import numpy as np
import pandas as pd

from fraud.replay import replay


def test_replay_structure_and_monotonicity():
    df = pd.DataFrame(
        {
            "Time": [0.0, 1800.0, 3600.0, 7200.0],
            "Amount": [10.0, 50.0, 100.0, 20.0],
            "Class": [0, 1, 0, 1],
        }
    )
    scores = np.array([0.1, 0.8, 0.2, 0.9])
    r = replay(df, scores, thr=0.5, fp_cost=5.0)

    # Columns required
    for col in [
        "Time",
        "Amount",
        "Class",
        "alert",
        "hour",
        "cost_model",
        "cost_none",
        "cum_model",
        "cum_none",
    ]:
        assert col in r.columns

    assert len(r) == len(df)
    # Cumulative costs must be non-decreasing
    assert np.all(np.diff(r["cum_model"]) >= 0)
    assert np.all(np.diff(r["cum_none"]) >= 0)
    # Fraud at index 1 is caught (score 0.8 >= 0.5) -> cost_model is 0, cost_none is 50
    assert r.loc[1, "cost_model"] == 0.0
    assert r.loc[1, "cost_none"] == 50.0
