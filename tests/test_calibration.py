import numpy as np

from fraud.calibration import (
    amount_aware_threshold,
    apply_calibrator,
    brier,
    capacity_constrained_flags,
    fit_isotonic,
    fit_platt,
)


def test_brier_perfect_and_worst():
    y = np.array([1, 0, 1, 0])
    perfect_s = np.array([1.0, 0.0, 1.0, 0.0])
    worst_s = np.array([0.0, 1.0, 0.0, 1.0])

    assert brier(y, perfect_s) == 0.0
    assert brier(y, worst_s) == 1.0


def test_platt_and_isotonic_monotonic():
    s_val = np.linspace(0.1, 0.9, 50)
    y_val = (s_val > 0.5).astype(int)

    platt = fit_platt(y_val, s_val)
    iso = fit_isotonic(y_val, s_val)

    test_s = np.array([0.2, 0.4, 0.6, 0.8])
    p_platt = apply_calibrator(platt, test_s)
    p_iso = apply_calibrator(iso, test_s)

    # Probabilities must be strictly bounded in [0, 1] and monotonic
    assert np.all((p_platt >= 0.0) & (p_platt <= 1.0))
    assert np.all((p_iso >= 0.0) & (p_iso <= 1.0))
    assert np.all(np.diff(p_platt) > 0)
    assert np.all(np.diff(p_iso) >= 0)


def test_amount_aware_rule():
    p = np.array([0.1, 0.5, 0.01])
    amt = np.array([100.0, 10.0, 200.0])
    # expected fraud = p * amt = [10.0, 5.0, 2.0]
    # at fp_cost = 5.0: [10.0 > 5, 5.0 > 5 (False), 2.0 > 5 (False)]
    flag = amount_aware_threshold(p, amt, fp_cost=5.0)
    assert np.array_equal(flag, [True, False, False])


def test_capacity_constrained_flags():
    s = np.array([0.9, 0.8, 0.7, 0.95, 0.6])
    hours = np.array([0, 0, 0, 1, 1])
    # Cap 1 per hour: hour 0 top is 0.9 (idx 0), hour 1 top is 0.95 (idx 3)
    flags = capacity_constrained_flags(s, n_per_hour=1, hour_indices=hours)
    assert np.array_equal(flags, [True, False, False, True, False])
