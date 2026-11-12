"""Cost-based evaluation and sensitivity analysis."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from fraud.config import DEFAULT_FP_COST, RESULTS_DIR
from fraud.cost import best_threshold, flag_nothing_cost, total_cost

d = np.load(RESULTS_DIR / "scores.npz")
yva, yte = d["yva"], d["yte"]
ava, ate = d["amt_va"], d["amt_te"]
names = ["logreg", "iforest", "xgb", "xgb_smote"]
prob_models = {"logreg", "xgb", "xgb_smote"}

FP = DEFAULT_FP_COST
base = flag_nothing_cost(yte, ate)
print(f"flag-nothing cost on test: ${base:.2f}")


def f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[np.argmax(f1)])


def at(y, s, a, t, fp):
    pred = s >= t
    c = total_cost(y, pred, a, fp)
    return dict(
        cost=round(c, 2),
        per_1000_tx=round(c / len(y) * 1000, 2),
        alerts=int(pred.sum()),
        caught=int((pred & (y == 1)).sum()),
    )


rows = []
for n in names:
    sv, st = d[f"{n}_va"], d[f"{n}_te"]
    t_cost, _ = best_threshold(yva, sv, ava, FP)
    t_f1 = f1_threshold(yva, sv)
    variants = {"cost_optimal": t_cost, "f1_optimal": t_f1}
    if n in prob_models:
        variants["default_0.5"] = 0.5
    for label, t in variants.items():
        r = at(yte, st, ate, t, FP)
        r.update(
            model=n,
            rule=label,
            threshold=round(t, 4),
            pct_loss_avoided=round(100 * (1 - r["cost"] / base), 1),
        )
        rows.append(r)

res = pd.DataFrame(rows)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
res.to_csv(RESULTS_DIR / "cost_results.csv", index=False)
print("\n=== Cost results ===")
print(res.to_string(index=False))

# save chosen threshold for replay
t_xgb, _ = best_threshold(yva, d["xgb_va"], ava, FP)
json.dump(
    {"model": "xgb", "threshold": t_xgb, "fp_cost": FP},
    open(RESULTS_DIR / "chosen_threshold.json", "w"),
)
print(f"\nChosen XGB threshold: {t_xgb:.4f}")

# sensitivity: vary FP cost
sens = []
for fp in (1, 2, 5, 10, 20, 50):
    for n in names:
        t, _ = best_threshold(yva, d[f"{n}_va"], ava, fp)
        r = at(yte, d[f"{n}_te"], ate, t, fp)
        sens.append(dict(fp_cost=fp, model=n, threshold=round(t, 4), **r))

sens_df = pd.DataFrame(sens)
sens_df.to_csv(RESULTS_DIR / "cost_sensitivity.csv", index=False)
print("\n=== Cost sensitivity (pivot by model) ===")
print(sens_df.pivot(index="fp_cost", columns="model", values="cost"))
print("\n=== XGB sensitivity ===")
print(sens_df[sens_df.model == "xgb"][["fp_cost", "threshold", "alerts", "caught"]])
