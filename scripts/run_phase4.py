"""Phase 4: Feature Engineering Ablation + Extended Model Comparison.

Steps:
  1. Ablation: compare val PR-AUC with FEATS_BASE vs FEATS_ENG on LR and XGBoost.
     Keeps engineering only if it improves val PR-AUC.
  2. Extended comparison: LightGBM and Random Forest on the winning feature set.
  3. Rank-average ensemble: XGBoost + LightGBM scores (supervised only).
  4. Write reports/results/ablation.csv and reports/results/phase4_comparison.csv.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud import models
from fraud.config import MODELS_DIR, RESULTS_DIR
from fraud.data import load
from fraud.evaluate import summary
from fraud.features import FEATS_BASE, FEATS_ENG, add_features
from fraud.split import time_split

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# Load and split
# ---------------------------------------------------------------
df_raw = load()
df_eng = add_features(df_raw)  # adds hour_of_day, log_amount

tr_raw, va_raw, te_raw = time_split(df_raw)
tr_eng, va_eng, te_eng = time_split(df_eng)

ytr = tr_raw["Class"]
yva = va_raw["Class"]
yte = te_raw["Class"]

# ---------------------------------------------------------------
# 1. Feature ablation on validation set
# ---------------------------------------------------------------
print("=== 1. Feature ablation (val PR-AUC) ===")

ablation_rows = []

for feat_label, feats, tr_df, va_df in [
    ("base", FEATS_BASE, tr_raw, va_raw),
    ("engineered", FEATS_ENG, tr_eng, va_eng),
]:
    Xtr_a = tr_df[feats]
    Xva_a = va_df[feats]

    # Logistic Regression
    lr_a = models.logreg(feats).fit(Xtr_a, ytr)
    lr_ap = average_precision_score(yva, lr_a.predict_proba(Xva_a)[:, 1])

    # XGBoost (fixed default params — no retuning to avoid leakage)
    xgb_params_default = dict(
        n_estimators=344,
        max_depth=7,
        learning_rate=0.1463,
        subsample=0.7823,
        colsample_bytree=0.8291,
        min_child_weight=2,
        reg_lambda=9.2705,
    )
    xgb_a = models.xgb(xgb_params_default, ytr, feats=feats).fit(Xtr_a, ytr)
    xgb_ap = average_precision_score(yva, xgb_a.predict_proba(Xva_a)[:, 1])

    ablation_rows.append(
        {
            "features": feat_label,
            "logreg_val_pr_auc": round(lr_ap, 4),
            "xgb_val_pr_auc": round(xgb_ap, 4),
        }
    )
    print(f"  {feat_label:12s}  logreg={lr_ap:.4f}  xgb={xgb_ap:.4f}")

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(RESULTS_DIR / "ablation.csv", index=False)
print(ablation_df.to_string(index=False))

# Decision: pick feature set that gives higher XGBoost val PR-AUC
base_ap = ablation_df.loc[ablation_df.features == "base", "xgb_val_pr_auc"].iloc[0]
eng_ap = ablation_df.loc[ablation_df.features == "engineered", "xgb_val_pr_auc"].iloc[0]

if eng_ap > base_ap:
    print(f"\nEngineered features improve val PR-AUC by {eng_ap - base_ap:.4f} → using FEATS_ENG")
    USE_ENG = True
    FEATS = FEATS_ENG
    tr_use, va_use, te_use = tr_eng, va_eng, te_eng
else:
    print(
        f"\nEngineered features do not improve val PR-AUC ({eng_ap:.4f} vs {base_ap:.4f}) → keeping FEATS_BASE"
    )
    USE_ENG = False
    FEATS = FEATS_BASE
    tr_use, va_use, te_use = tr_raw, va_raw, te_raw

Xtr = tr_use[FEATS]
Xva = va_use[FEATS]
Xte = te_use[FEATS]

# ---------------------------------------------------------------
# 2. Extended model comparison (same feature set, same split)
# ---------------------------------------------------------------
print("\n=== 2. Extended model comparison ===")

scores = {}

# Load previously tuned XGBoost params from existing model comparison
xgb_params = dict(
    n_estimators=344,
    max_depth=7,
    learning_rate=0.1463,
    subsample=0.7823,
    colsample_bytree=0.8291,
    min_child_weight=2,
    reg_lambda=9.2705,
)

# XGBoost (reference)
print("--- XGBoost ---")
xgb_m = models.xgb(xgb_params, ytr, feats=FEATS).fit(Xtr, ytr)
scores["xgb"] = (xgb_m.predict_proba(Xva)[:, 1], xgb_m.predict_proba(Xte)[:, 1])
print(f"  val PR-AUC: {average_precision_score(yva, scores['xgb'][0]):.4f}")

# LightGBM
print("--- LightGBM ---")
lgbm_m = models.lgbm(y=ytr, feats=FEATS).fit(Xtr, ytr)
scores["lgbm"] = (lgbm_m.predict_proba(Xva)[:, 1], lgbm_m.predict_proba(Xte)[:, 1])
print(f"  val PR-AUC: {average_precision_score(yva, scores['lgbm'][0]):.4f}")

# Random Forest
print("--- Random Forest ---")
rf_m = models.random_forest(FEATS).fit(Xtr, ytr)
scores["rf"] = (rf_m.predict_proba(Xva)[:, 1], rf_m.predict_proba(Xte)[:, 1])
print(f"  val PR-AUC: {average_precision_score(yva, scores['rf'][0]):.4f}")

# Rank-average ensemble: XGB + LightGBM (both supervised)
print("--- Rank-Average Ensemble (XGB + LightGBM) ---")
ens_va = models.rank_avg_ensemble([scores["xgb"][0], scores["lgbm"][0]])
ens_te = models.rank_avg_ensemble([scores["xgb"][1], scores["lgbm"][1]])
scores["ensemble_xgb_lgbm"] = (ens_va, ens_te)
print(f"  val PR-AUC: {average_precision_score(yva, ens_va):.4f}")

# ---------------------------------------------------------------
# 3. Build comparison table
# ---------------------------------------------------------------
rows = []
for name, (sv, st) in scores.items():
    r = {"model": name, "val_pr_auc": round(average_precision_score(yva, sv), 4)}
    r.update({f"test_{k}": v for k, v in summary(yte, st).items()})
    rows.append(r)

comp_df = pd.DataFrame(rows)
comp_df["test_no_skill_pr_auc"] = round(float(yte.mean()), 5)
comp_df["feature_set"] = "engineered" if USE_ENG else "base"
comp_df.to_csv(RESULTS_DIR / "phase4_comparison.csv", index=False)

print("\n=== Phase 4 Model Comparison ===")
print(
    comp_df[
        [
            "model",
            "val_pr_auc",
            "test_pr_auc",
            "test_recall@precision80",
            "test_recall@fpr0.1%",
            "test_roc_auc",
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------------
# 4. Save best Phase 4 model (highest val PR-AUC) and its scores
# ---------------------------------------------------------------
best_name = max(scores, key=lambda k: average_precision_score(yva, scores[k][0]))
print(f"\nBest Phase 4 model by val PR-AUC: {best_name}")

# Save LightGBM for Streamlit app (smaller than RF, interpretable with SHAP)
joblib.dump(lgbm_m, MODELS_DIR / "lgbm.joblib")
print("Saved lgbm.joblib")

# Extend scores.npz with Phase 4 models
existing = dict(np.load(RESULTS_DIR / "scores.npz", allow_pickle=False))
for k, (sv, st) in scores.items():
    if k not in ("xgb",):  # xgb already in scores.npz
        existing[f"{k}_va"] = sv
        existing[f"{k}_te"] = st
np.savez(RESULTS_DIR / "scores.npz", **existing)
print("Updated scores.npz with phase4 models")
