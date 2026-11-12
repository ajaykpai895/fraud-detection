"""SHAP explainability and error analysis for the XGBoost model."""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

from fraud.config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR
from fraud.data import load
from fraud.features import FEATS
from fraud.split import time_split

model_path = MODELS_DIR / "xgb.joblib"
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}. Run scripts/run_modeling.py first.")
model = joblib.load(model_path)
_, _, te = time_split(load())
X = te[FEATS]

print("Computing SHAP values...")
explainer = shap.TreeExplainer(model)
sv = explainer(X)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# SHAP summary plot
shap.summary_plot(sv.values, X, show=False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved shap_summary.png")

# individual fraud cases (first 3 actual frauds in test set)
fraud_idx = np.where(te.Class.values == 1)[0]
for k, i in enumerate(fraud_idx[:3]):
    shap.plots.waterfall(sv[i], show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"shap_case_{k}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved shap_case_{k}.png  (amount=${te.iloc[i].Amount:.2f})")

# error analysis: missed frauds vs caught frauds
thr_cfg = json.load(open(RESULTS_DIR / "chosen_threshold.json"))
thr = thr_cfg["threshold"]
score = model.predict_proba(X)[:, 1]
missed = te[(te.Class == 1) & (score < thr)]
caught = te[(te.Class == 1) & (score >= thr)]

print(f"\nAt threshold {thr:.4f}:")
print(f"  caught: {len(caught)}, missed: {len(missed)}")
print("\nAmount distribution, missed frauds:")
print(missed.Amount.describe())
print("\nAmount distribution, caught frauds:")
print(caught.Amount.describe())

# hour-of-window distribution
missed_hours = ((missed.Time - te.Time.min()) // 3600).astype(int)
caught_hours = ((caught.Time - te.Time.min()) // 3600).astype(int)
print("\nMissed frauds per hour (top 5 hours):")
print(missed_hours.value_counts().head())
print("\nCaught frauds per hour (top 5 hours):")
print(caught_hours.value_counts().head())
