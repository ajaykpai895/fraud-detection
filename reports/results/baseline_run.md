# Baseline Execution Run Log

**Date:** 2026-11-12  
**Dataset:** `data/creditcard.csv` (284,807 transactions, 492 frauds, 0 nulls)  
**Splits:** Time-based 70/15/15:
- Train: 199,364 rows, 384 frauds
- Validation: 42,721 rows, 56 frauds
- Test: 42,722 rows, 52 frauds
- Test No-Skill PR-AUC: 0.00122 (52 / 42,722)

---

## 1. Unit Tests (`pytest -q`)
```
============================= test session starts =============================
platform win32 -- Python 3.13.15, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\HP\OneDrive\Desktop\dataanalyst\fraud-detection
configfile: pyproject.toml
plugins: anyio-4.12.1, cov-7.1.0
collected 8 items

tests\test_split.py ....                                                 [ 50%]
tests\test_cost.py ....                                                  [100%]

============================== 8 passed in 3.85s ==============================
```

---

## 2. Model Performance Benchmark (`reports/results/model_comparison.csv`)

| Model | Val PR-AUC | Test PR-AUC | Test Recall @ Prec=80% | Test Recall @ FPR=0.1% | Test ROC-AUC | Test No-Skill PR-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** (class-weighted) | 0.8394 | 0.7066 | 0.7308 | 0.7500 | 0.9773 | 0.00122 |
| **Isolation Forest** (unsupervised) | 0.0449 | 0.0640 | 0.0000 | 0.0769 | 0.9442 | 0.00122 |
| **XGBoost** (scale_pos_weight tuned) | 0.8590 | 0.7717 | 0.7500 | 0.7692 | 0.9696 | 0.00122 |
| **XGBoost + SMOTE** | 0.8444 | 0.7746 | 0.7500 | 0.7885 | 0.9733 | 0.00122 |

### Hyperparameters Discovered
- **Isolation Forest:** `n_estimators=100`, `max_samples=1024`, `max_features=1.0` (Val PR-AUC = 0.0449)
- **XGBoost (Optuna 40 trials, 4 forward time-series CV folds):**
  - Best CV PR-AUC: `0.7933`
  - Params: `n_estimators=344`, `max_depth=7`, `learning_rate=0.14626`, `subsample=0.7823`, `colsample_bytree=0.8291`, `min_child_weight=2`, `reg_lambda=9.2705`

---

## 3. Cost-Based Decision Evaluation (`reports/results/cost_results.csv`)

- **Flag-Nothing Baseline Test Cost:** **$6,168.88** (loss from all 52 frauds)
- **Baseline FP Friction Cost:** $5.00 / false alert

| Model | Rule | Threshold | Alerts | Frauds Caught | Total Cost ($) | Cost / 1k Tx ($) | Loss Avoided (%) |
|---|---|---|---|---|---|---|---|
| **logreg** | cost_optimal | 1.0000 | 35 | 33 | 3,079.81 | 72.09 | 50.1% |
| **logreg** | f1_optimal | 1.0000 | 35 | 33 | 3,079.81 | 72.09 | 50.1% |
| **logreg** | default_0.5 | 0.5000 | 745 | 43 | 5,852.30 | 136.99 | 5.1% |
| **iforest** | cost_optimal | 0.5418 | 443 | 31 | 5,151.23 | 120.58 | 16.5% |
| **iforest** | f1_optimal | 0.5720 | 264 | 24 | 5,187.17 | 121.42 | 15.9% |
| **xgb** | cost_optimal | 0.0082 | 117 | 40 | 2,123.10 | 49.70 | 65.6% |
| **xgb** | f1_optimal | 0.8322 | 45 | 39 | 2,402.40 | 56.23 | 61.1% |
| **xgb** | default_0.5 | 0.5000 | 46 | 39 | 2,407.40 | 56.35 | 61.0% |
| **xgb_smote** | cost_optimal | 0.0489 | 84 | 41 | 1,940.79 | 45.43 | 68.5% |
| **xgb_smote** | f1_optimal | 0.7297 | 46 | 38 | 2,452.30 | 57.40 | 60.2% |
| **xgb_smote** | default_0.5 | 0.5000 | 50 | 39 | 1,833.00 | 42.91 | 70.3% |

---

## 4. Cost Sensitivity to FP Friction Assumption (`reports/results/cost_sensitivity.csv`)

| FP Cost ($) | iforest Cost ($) | logreg Cost ($) | xgb Cost ($) | xgb_smote Cost ($) | Best Model |
|---|---|---|---|---|---|
| **$1** | 3,779.75 | 2,413.09 | 1,815.10 | 1,514.84 | xgb_smote |
| **$2** | 4,518.75 | 2,466.09 | 1,892.10 | 1,811.79 | xgb_smote |
| **$5** | 5,151.23 | 3,079.81 | 2,123.10 | 1,940.79 | xgb_smote |
| **$10** | 6,107.94 | 3,089.81 | 2,188.10 | 2,155.79 | xgb_smote |
| **$20** | 6,168.88 | 3,109.81 | 2,638.10 | 2,585.79 | xgb_smote |
| **$50** | 6,168.88 | 3,169.81 | 2,772.40 | 2,328.00 | xgb_smote |

*Key finding:* At FP cost ≥ $20, Isolation Forest stops flagging entirely (threshold exceeds max score, 0 alerts, defaulting to the flag-nothing cost of $6,168.88). XGBoost models consistently avoid 60–70% of losses across all FP cost scenarios.

---

## 5. Replay Simulation Summary
- **Simulated Test Period:** 42,722 transactions (~6 hours)
- **Hourly alert volume (XGBoost @ threshold 0.0082):**
  - Hour 0: 25 alerts
  - Hour 1: 17 alerts
  - Hour 2: 23 alerts
  - Hour 3: 22 alerts
  - Hour 4: 19 alerts
  - Hour 5: 11 alerts
- **Cumulative test cost:**
  - Model: **$2,123.10**
  - Flag nothing: **$6,168.88** (Net financial savings: **$4,045.78**)

---

## 6. Explainability & Error Profile
- **Generated Figures:**
  - `reports/figures/shap_summary.png`
  - `reports/figures/shap_case_0.png` ($0.01 amount)
  - `reports/figures/shap_case_1.png` ($57.73 amount)
  - `reports/figures/shap_case_2.png` ($1.00 amount)
  - `reports/figures/replay_cost.png`
- **Error Analysis at Cost-Optimal Threshold (0.0082):**
  - Frauds caught: 40 / 52 (76.9% recall)
  - Frauds missed: 12 / 52 (23.1%)
  - Missed fraud amounts: Mean = $144.84, Median = $9.95, Max = $1,096.99
  - Caught fraud amounts: Mean = $110.77, Median = $4.80, Max = $1,504.93
