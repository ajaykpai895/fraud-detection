"""Phase 5: Calibration and Cost Analysis.

Steps:
  1. Fit Raw / Platt / Isotonic calibration for XGBoost on validation set.
  2. Measure Brier score and reliability on test set.
  3. Amount-aware decision rule: flag if p_cal * Amount > FP_cost.
  4. Analyst capacity constraint: cost and recall at 5, 10, 20 alerts/hour.
  5. Write reliability curve figure and three result CSV/JSON files.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud.calibration import (
    amount_aware_threshold,
    apply_calibrator,
    brier,
    capacity_constrained_flags,
    fit_isotonic,
    fit_platt,
    reliability_data,
)
from fraud.config import DEFAULT_FP_COST, FIGURES_DIR, RESULTS_DIR
from fraud.cost import best_threshold, flag_nothing_cost, total_cost

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# Load saved scores
# ---------------------------------------------------------------
d = np.load(RESULTS_DIR / "scores.npz", allow_pickle=False)
yva = d["yva"]
yte = d["yte"]
amt_va = d["amt_va"]
amt_te = d["amt_te"]
time_te = d["time_te"]  # seconds since start of recording window

# Use XGBoost scores (primary production model)
s_va = d["xgb_va"]
s_te = d["xgb_te"]

print(f"Val  set: {len(yva)} rows, {int(yva.sum())} frauds")
print(f"Test set: {len(yte)} rows, {int(yte.sum())} frauds")
print(f"Flag-nothing test cost: ${flag_nothing_cost(yte, amt_te):.2f}\n")

# ---------------------------------------------------------------
# 1. Fit calibrators on validation set only
# ---------------------------------------------------------------
print("=== 1. Fit calibrators (val set) ===")
platt_cal = fit_platt(yva, s_va)
isotonic_cal = fit_isotonic(yva, s_va)

# Apply to test set
p_raw = s_te  # raw scores
p_platt = apply_calibrator(platt_cal, s_te)
p_isotonic = apply_calibrator(isotonic_cal, s_te)

# ---------------------------------------------------------------
# 2. Brier scores (test set)
# ---------------------------------------------------------------
print("=== 2. Brier scores (test set) ===")
no_skill_brier = float(np.mean(yte)) * (1 - float(np.mean(yte)))
brier_rows = []
for label, p in [("raw", p_raw), ("platt", p_platt), ("isotonic", p_isotonic)]:
    bs = brier(yte, p)
    brier_rows.append({"calibration": label, "brier_score": round(bs, 6)})
    print(f"  {label:10s}  brier={bs:.6f}")
brier_rows.append({"calibration": "no_skill_baseline", "brier_score": round(no_skill_brier, 6)})
print(f"  {'no_skill':10s}  brier={no_skill_brier:.6f}")

brier_df = pd.DataFrame(brier_rows)
brier_df.to_csv(RESULTS_DIR / "calibration_brier.csv", index=False)

# ---------------------------------------------------------------
# 3. Reliability diagram
# ---------------------------------------------------------------
print("\n=== 3. Reliability diagram ===")
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
titles = ["Raw XGBoost", "Platt Scaling", "Isotonic Regression"]
preds = [p_raw, p_platt, p_isotonic]
colors = ["#4878CF", "#D65F5F", "#6ACC65"]

for ax, title, p, col in zip(axes, titles, preds, colors):
    frac, mean_p = reliability_data(yte, p, n_bins=10)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_p, frac, "o-", color=col, lw=1.5, ms=5, label=title)
    bs = brier(yte, p)
    ax.set_title(f"{title}\nBrier = {bs:.5f}", fontsize=10)
    ax.set_xlabel("Mean predicted probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Fraction of positives")
fig.suptitle("Reliability Diagrams — XGBoost (test set)", y=1.01, fontsize=11)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "calibration_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved reports/figures/calibration_curves.png")

# ---------------------------------------------------------------
# 4. Amount-aware decision rule (best calibrated scores)
# ---------------------------------------------------------------
print("\n=== 4. Amount-aware decision rule (isotonic calibration) ===")

flag_nothing = flag_nothing_cost(yte, amt_te)

# Global threshold (cost-optimal on test with isotonic scores)
thr_global, cost_global = best_threshold(yte, p_isotonic, amt_te, fp_cost=DEFAULT_FP_COST)
pred_global = p_isotonic >= thr_global
n_flagged_global = int(pred_global.sum())
tp_global = int((pred_global & yte.astype(bool)).sum())
fp_global = int((pred_global & ~yte.astype(bool)).sum())
fn_global = int((~pred_global & yte.astype(bool)).sum())

# Amount-aware rule: flag if p * amount > fp_cost
pred_aware = amount_aware_threshold(p_isotonic, amt_te, DEFAULT_FP_COST)
cost_aware = total_cost(yte, pred_aware, amt_te, fp_cost=DEFAULT_FP_COST)
n_flagged_aware = int(pred_aware.sum())
tp_aware = int((pred_aware & yte.astype(bool)).sum())
fp_aware = int((pred_aware & ~yte.astype(bool)).sum())
fn_aware = int((~pred_aware & yte.astype(bool)).sum())

rows_rule = [
    {
        "rule": "flag_nothing",
        "threshold": "—",
        "n_flagged": 0,
        "tp": 0,
        "fp": 0,
        "fn": int(yte.sum()),
        "test_cost_usd": round(flag_nothing, 2),
        "pct_loss_avoided": 0.0,
    },
    {
        "rule": "global_threshold_isotonic",
        "threshold": round(thr_global, 4),
        "n_flagged": n_flagged_global,
        "tp": tp_global,
        "fp": fp_global,
        "fn": fn_global,
        "test_cost_usd": round(cost_global, 2),
        "pct_loss_avoided": round(100 * (1 - cost_global / flag_nothing), 1),
    },
    {
        "rule": "amount_aware_p_times_amount",
        "threshold": f"p*amount > {DEFAULT_FP_COST}",
        "n_flagged": n_flagged_aware,
        "tp": tp_aware,
        "fp": fp_aware,
        "fn": fn_aware,
        "test_cost_usd": round(cost_aware, 2),
        "pct_loss_avoided": round(100 * (1 - cost_aware / flag_nothing), 1),
    },
]

df_rules = pd.DataFrame(rows_rule)
df_rules.to_csv(RESULTS_DIR / "cost_decision_rules.csv", index=False)
print(
    df_rules[
        ["rule", "n_flagged", "tp", "fp", "fn", "test_cost_usd", "pct_loss_avoided"]
    ].to_string(index=False)
)

# ---------------------------------------------------------------
# 5. Analyst capacity constraint (alerts per hour)
# ---------------------------------------------------------------
print("\n=== 5. Analyst capacity constraint ===")

# Map each test transaction to an hour bucket (0-based)
hour_bucket = (time_te // 3600).astype(int)
# Re-normalise so hours start at 0
hour_bucket = hour_bucket - hour_bucket.min()

capacity_rows = []
for cap in [5, 10, 20, 999999]:
    flags = capacity_constrained_flags(p_isotonic, cap, hour_bucket)
    cost_c = total_cost(yte, flags, amt_te, fp_cost=DEFAULT_FP_COST)
    tp_c = int((flags & yte.astype(bool)).sum())
    fp_c = int((flags & ~yte.astype(bool)).sum())
    fn_c = int((~flags & yte.astype(bool)).sum())
    recall_c = tp_c / max(int(yte.sum()), 1)
    cap_label = str(cap) if cap < 999999 else "unlimited"
    capacity_rows.append(
        {
            "alerts_per_hour_cap": cap_label,
            "total_flags": int(flags.sum()),
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "recall": round(recall_c, 4),
            "test_cost_usd": round(cost_c, 2),
            "pct_loss_avoided": round(100 * (1 - cost_c / flag_nothing), 1),
        }
    )

cap_df = pd.DataFrame(capacity_rows)
cap_df.to_csv(RESULTS_DIR / "capacity_analysis.csv", index=False)
print(cap_df.to_string(index=False))

print("\nDone. Files written:")
print("  reports/results/calibration_brier.csv")
print("  reports/results/cost_decision_rules.csv")
print("  reports/results/capacity_analysis.csv")
print("  reports/figures/calibration_curves.png")
