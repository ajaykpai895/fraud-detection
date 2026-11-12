from pathlib import Path

from fraud.config import (
    CV_SPLITS,
    DATA_PATH,
    DEFAULT_FP_COST,
    DEFAULT_TP_COST,
    FIGURES_DIR,
    MODELS_DIR,
    N_OPTUNA_TRIALS,
    RESULTS_DIR,
    ROOT,
    SEED,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)


def test_split_fractions_sum_to_one():
    total = TRAIN_FRAC + VAL_FRAC + TEST_FRAC
    assert abs(total - 1.0) < 1e-9, f"Splits do not sum to 1.0: {total}"


def test_positive_integers():
    assert SEED >= 0
    assert CV_SPLITS >= 2
    assert N_OPTUNA_TRIALS >= 1


def test_cost_defaults():
    assert DEFAULT_FP_COST > 0
    assert DEFAULT_TP_COST >= 0


def test_paths():
    assert isinstance(ROOT, Path)
    assert isinstance(DATA_PATH, Path)
    assert isinstance(RESULTS_DIR, Path)
    assert isinstance(FIGURES_DIR, Path)
    assert isinstance(MODELS_DIR, Path)
