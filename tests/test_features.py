import numpy as np
import pandas as pd

from fraud.features import FEATS, V, make_pre


def test_feats_length_and_order():
    assert len(FEATS) == 30
    assert FEATS[0] == "Time"
    assert FEATS[1] == "Amount"
    assert len(V) == 28


def test_preprocessor_preserves_v_features():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((100, 30))
    df = pd.DataFrame(data, columns=FEATS)

    pre = make_pre().fit(df)
    transformed = pre.transform(df)

    assert transformed.shape == (100, 30)
    # The remainder features (V1..V28) start at column index 2
    # and must be identical to original V features
    np.testing.assert_allclose(transformed[:, 2:], df[V].values)
