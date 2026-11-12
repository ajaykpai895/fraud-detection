"""Phase 3: Rigorous Evaluation
- 1,000 bootstrap confidence intervals for PR-AUC and cost
- Paired bootstrap difference (XGBoost vs next best model)
- Rolling-origin (expanding window) temporal evaluation and spread analysis
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud.config import DEFAULT_FP_COST, FIGURES_DIR, RESULTS_DIR, SEED
from fraud.data import load
from fraud.evaluate import bootstrap_cost, bootstrap_paired_diff, bootstrap_pr_auc
from fraud.rolling import run_rolling_evaluation

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load existing test scores
scores_path = RESULTS_DIR / "scores.npz"
if not scores_path.exists():
    raise FileNotFoundError("scores.npz not found. Run scripts/run_modeling.py first.")

d = np.load(scores_path)
yte = d["yte"]
ate = d["amt_te"]
names = ["logreg", "iforest", "xgb", "xgb_smote"]

print(f"Loaded test set: {len(yte)} rows, {int(yte.sum())} frauds")

# ---------------------------------------------------------
# 1. Bootstrap PR-AUC Confidence Intervals (1,000 resamples)
# ---------------------------------------------------------
print("\n--- 1. Computing 1,000-sample Bootstrap PR-AUC 95% CIs ---")
base_comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")

ci_rows = []
for n in names:
    st = d[f"{n}_te"]
    b_res = bootstrap_pr_auc(yte, st, n_bootstraps=1000, seed=SEED)
    ci_rows.append(
        {
            "model": n,
            "test_pr_auc": b_res["point"],
            "ci_95_lower": b_res["ci_lower"],
            "ci_95_upper": b_res["ci_upper"],
            "ci_std": b_res["std"],
        }
    )

# Add no-skill baseline
b_noskill = bootstrap_pr_auc(
    yte, np.full_like(yte, yte.mean(), dtype=float), n_bootstraps=1000, seed=SEED
)
ci_rows.append(
    {
        "model": "no_skill",
        "test_pr_auc": b_noskill["point"],
        "ci_95_lower": b_noskill["ci_lower"],
        "ci_95_upper": b_noskill["ci_upper"],
        "ci_std": b_noskill["std"],
    }
)

df_ci = pd.DataFrame(ci_rows)
# Merge with base comparison
merged_comp = pd.merge(
    base_comp, df_ci[["model", "ci_95_lower", "ci_95_upper", "ci_std"]], on="model", how="left"
)
merged_comp.to_csv(RESULTS_DIR / "model_comparison_with_ci.csv", index=False)
print(merged_comp.to_string(index=False))

# ---------------------------------------------------------
# 2. Bootstrap Cost Confidence Intervals (1,000 resamples)
# ---------------------------------------------------------
print("\n--- 2. Computing 1,000-sample Bootstrap Cost 95% CIs ---")
cost_df = pd.read_csv(RESULTS_DIR / "cost_results.csv")
cost_ci_rows = []

for _, row in cost_df.iterrows():
    m = row["model"]
    thr = row["threshold"]
    st = d[f"{m}_te"]
    pred = st >= thr
    b_c = bootstrap_cost(yte, pred, ate, fp_cost=DEFAULT_FP_COST, n_bootstraps=1000, seed=SEED)
    cost_ci_rows.append(
        {
            "model": m,
            "rule": row["rule"],
            "cost": b_c["point"],
            "cost_ci_lower": b_c["ci_lower"],
            "cost_ci_upper": b_c["ci_upper"],
            "cost_std": b_c["std"],
        }
    )

cost_ci_df = pd.DataFrame(cost_ci_rows)
merged_cost = pd.merge(
    cost_df,
    cost_ci_df[["model", "rule", "cost_ci_lower", "cost_ci_upper", "cost_std"]],
    on=["model", "rule"],
)
merged_cost.to_csv(RESULTS_DIR / "cost_results_with_ci.csv", index=False)
print(
    merged_cost[["model", "rule", "threshold", "cost", "cost_ci_lower", "cost_ci_upper"]].to_string(
        index=False
    )
)

# ---------------------------------------------------------
# 3. Paired Bootstrap Comparison (XGBoost vs Next Best)
# ---------------------------------------------------------
print("\n--- 3. Paired Bootstrap Difference (XGBoost vs Alternatives) ---")
paired_results = {}

# Compare xgb vs logreg
diff_xgb_lr = bootstrap_paired_diff(yte, d["xgb_te"], d["logreg_te"], n_bootstraps=1000, seed=SEED)
paired_results["xgb_vs_logreg"] = diff_xgb_lr
print("XGBoost vs Logistic Regression:")
print(f"  Delta PR-AUC (XGB - LR): {diff_xgb_lr['point_diff']:.4f}")
print(f"  95% CI: [{diff_xgb_lr['ci_lower']:.4f}, {diff_xgb_lr['ci_upper']:.4f}]")
print(
    f"  Empirical p-value: {diff_xgb_lr['p_value']:.4f} (Significant: {diff_xgb_lr['significant']})"
)

# Compare xgb vs xgb_smote
diff_xgb_smote = bootstrap_paired_diff(
    yte, d["xgb_te"], d["xgb_smote_te"], n_bootstraps=1000, seed=SEED
)
paired_results["xgb_vs_xgb_smote"] = diff_xgb_smote
print("\nXGBoost vs XGBoost + SMOTE:")
print(f"  Delta PR-AUC (XGB - SMOTE): {diff_xgb_smote['point_diff']:.4f}")
print(f"  95% CI: [{diff_xgb_smote['ci_lower']:.4f}, {diff_xgb_smote['ci_upper']:.4f}]")
print(
    f"  Empirical p-value: {diff_xgb_smote['p_value']:.4f} (Significant: {diff_xgb_smote['significant']})"
)

with open(RESULTS_DIR / "paired_test.json", "w") as f:
    json.dump(paired_results, f, indent=2)

# ---------------------------------------------------------
# 4. Rolling-Origin (Expanding Window) Temporal Evaluation
# ---------------------------------------------------------
print("\n--- 4. Running Rolling-Origin Temporal Evaluation (4 periods) ---")
df_full = load()
rolling_df, rolling_summary = run_rolling_evaluation(df_full)

rolling_df.to_csv(RESULTS_DIR / "rolling_evaluation.csv", index=False)
rolling_summary.to_csv(RESULTS_DIR / "rolling_evaluation_summary.csv", index=False)

print("\nRolling Evaluation by Period:")
print(rolling_df.to_string(index=False))

print("\nStability Across Time Periods (Mean & Spread):")
print(rolling_summary.to_string(index=False))

# Plot rolling stability
fig, ax = plt.subplots(figsize=(8, 4.5))
periods = rolling_df["period"].tolist()
x = np.arange(len(periods))
width = 0.2

ax.bar(x - 1.5 * width, rolling_df["no_skill_pr_auc"], width, label="No Skill", color="#9E9E9E")
ax.bar(
    x - 0.5 * width, rolling_df["iforest_pr_auc"], width, label="Isolation Forest", color="#6ACC65"
)
ax.bar(
    x + 0.5 * width,
    rolling_df["logreg_pr_auc"],
    width,
    label="Logistic Regression",
    color="#4878CF",
)
ax.bar(x + 1.5 * width, rolling_df["xgb_pr_auc"], width, label="XGBoost", color="#D65F5F")

ax.set_ylabel("PR-AUC")
ax.set_title("Model Stability Over Rolling Temporal Periods (Expanding Window)")
ax.set_xticks(x)
ax.set_xticklabels(periods)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "rolling_eval.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved reports/figures/rolling_eval.png")
