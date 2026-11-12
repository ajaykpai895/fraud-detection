"""Train all models and write comparison table + score arrays to reports/results/.

Uses the FEATS_BASE (no engineering) feature set. run_phase4.py runs the
ablation and extended model comparison on top of this.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud import models
from fraud.config import MODELS_DIR, N_OPTUNA_TRIALS, RESULTS_DIR
from fraud.data import load
from fraud.evaluate import summary
from fraud.features import FEATS_BASE
from fraud.split import cv_folds, time_split

df = load()
tr, va, te = time_split(df)

for n, d in [("train", tr), ("val", va), ("test", te)]:
    print(f"{n}: {len(d)} rows, {int(d.Class.sum())} frauds")

Xtr, Xva, Xte = tr[FEATS_BASE], va[FEATS_BASE], te[FEATS_BASE]
ytr, yva, yte = tr["Class"], va["Class"], te["Class"]

folds = cv_folds(len(tr))
for k, (a, b) in enumerate(folds):
    print(f"fold {k}: train frauds {int(ytr.iloc[a].sum())}, val frauds {int(ytr.iloc[b].sum())}")

scores = {}

# 1. logistic regression baseline
print("\n--- Logistic Regression ---")
lr = models.logreg(FEATS_BASE).fit(Xtr, ytr)
scores["logreg"] = (lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1])
print("val pr_auc:", average_precision_score(yva, scores["logreg"][0]))

# 2. isolation forest — fitted without labels; tune n_estimators, max_samples, max_features
print("\n--- Isolation Forest grid search ---")
best = (-1, None, None)
for n_est in (100, 300):
    for ms in (256, 1024):
        for mf in (0.5, 1.0):
            m = models.iforest(n_estimators=n_est, max_samples=ms, max_features=mf).fit(Xtr)
            ap = average_precision_score(yva, -m.score_samples(Xva))
            print(f"  n={n_est} ms={ms} mf={mf}  val pr_auc={ap:.4f}")
            if ap > best[0]:
                best = (ap, m, dict(n_estimators=n_est, max_samples=ms, max_features=mf))

print("best params:", best[2])
if_model = best[1]
scores["iforest"] = (-if_model.score_samples(Xva), -if_model.score_samples(Xte))

# 3. xgboost with class weighting, tuned with optuna on time-aware CV
print("\n--- XGBoost Optuna tuning ---")
study = models.tune_xgb(Xtr, ytr, folds, n_trials=N_OPTUNA_TRIALS)
print("best cv pr-auc:", study.best_value)
print("best params:", study.best_params)
xgb_model = models.xgb(study.best_params, ytr).fit(Xtr, ytr)
scores["xgb"] = (
    xgb_model.predict_proba(Xva)[:, 1],
    xgb_model.predict_proba(Xte)[:, 1],
)
print("val pr_auc:", average_precision_score(yva, scores["xgb"][0]))

# 4. xgboost with SMOTE — same hyperparameters, test whether resampling helps
print("\n--- XGBoost + SMOTE ---")
sm = models.xgb_smote(study.best_params).fit(Xtr, ytr)
scores["xgb_smote"] = (sm.predict_proba(Xva)[:, 1], sm.predict_proba(Xte)[:, 1])
print("val pr_auc:", average_precision_score(yva, scores["xgb_smote"][0]))

# comparison table
rows = []
for name, (sv, st) in scores.items():
    r = {"model": name, "val_pr_auc": summary(yva, sv)["pr_auc"]}
    r.update({f"test_{k}": v for k, v in summary(yte, st).items()})
    rows.append(r)

tab = pd.DataFrame(rows).round(4)
tab["test_no_skill_pr_auc"] = round(float(yte.mean()), 5)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
tab.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
print("\n=== Model comparison ===")
print(tab.to_string(index=False))

# save scores for cost analysis and notebook 02
np.savez(
    RESULTS_DIR / "scores.npz",
    yva=yva.values,
    yte=yte.values,
    amt_va=va.Amount.values,
    amt_te=te.Amount.values,
    time_te=te.Time.values,
    **{f"{k}_va": v[0] for k, v in scores.items()},
    **{f"{k}_te": v[1] for k, v in scores.items()},
)
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(xgb_model, MODELS_DIR / "xgb.joblib")
print("Saved scores.npz and xgb.joblib")
