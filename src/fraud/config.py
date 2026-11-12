"""Central configuration for paths, random seeds, split ratios, and cost defaults."""

from pathlib import Path

# Repository root and key directory paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "creditcard.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"

# Reproducibility
SEED = 42

# Time-based splitting
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
CV_SPLITS = 4

# Modeling
N_OPTUNA_TRIALS = 40

# Cost evaluation assumptions (USD)
DEFAULT_FP_COST = 5.0
DEFAULT_TP_COST = 0.0

# Expected dataset signature
EXPECTED_ROWS = 284807
EXPECTED_COLS = 31
EXPECTED_FRAUDS = 492
