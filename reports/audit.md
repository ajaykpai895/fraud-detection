# Audit Report: Credit Card Fraud Detection Pipeline

**Auditor:** Senior Data Scientist  
**Scope:** Repository architecture, data pipeline, validation integrity, modeling methodology, cost formulation, explainability, tests, and reproducibility.  
**Dataset:** Real Kaggle Credit Card Fraud Detection (`mlg-ulb/creditcardfraud`, 284,807 transactions, 492 frauds, 0.1727% positive class).

---

## 1. Executive Summary & Hiring Manager Perspective

A senior hiring manager evaluating this portfolio project is looking for three core attributes:
1. **Methodological integrity:** Did the candidate understand class imbalance, temporal ordering, and asymmetric loss, or did they naively optimize accuracy on a randomized train/test split?
2. **Engineering rigor:** Is the repository modular, reproducible with a single command, free of leakage, and covered by sensible automated tests?
3. **Intellectual honesty:** Are limitations, negative results (e.g. SMOTE underperformance), and business realities (analyst review capacity, operational costs) acknowledged without marketing jargon?

The current project establishes a strong foundational architecture—temporal splitting, time-aware cross-validation, PR-AUC prioritization, and cost-curve optimization. However, several critical gaps and weaknesses weaken its impact from "promising student work" to "production-grade junior data scientist."

---

## 2. Detailed Findings: What Works, What is Weak, What is Missing

### A. What Works Well
- **Real dataset adherence:** Uses the authentic Kaggle dataset (284,807 rows, 492 frauds, 31 columns).
- **Time-based validation design:** Splits data strictly chronologically (70% train / 15% validation / 15% test) without temporal overlap between sets.
- **Forward-only cross-validation:** Uses `TimeSeriesSplit` (4 folds) ensuring training data strictly precedes validation data in all folds.
- **No data leakage in preprocessing:** `make_pre()` (`RobustScaler` on `Time` and `Amount`) is properly wrapped in scikit-learn / imbalanced-learn pipelines and fitted strictly on training data folds.
- **Problem framed around asymmetric cost:** Rejects accuracy and ROC-AUC as primary metrics. Defines explicit financial loss for false negatives ($\text{Amount}$) and operational friction for false positives ($\$5.00$ baseline).
- **Threshold optimization on validation:** Optimal decision thresholds are determined on validation sets and evaluated out-of-sample on the held-out test set.
- **Unit test foundation:** Automated tests in `tests/test_split.py` and `tests/test_cost.py` verify no temporal overlap, forward-only CV folds, scaler fit isolation, and hand-computed cost functions.

---

### B. What is Weak (Ranked by Hiring Impact)

#### 1. [HIGH IMPACT] Statistical Fragility of Held-Out Test Evaluation
- **Issue:** The held-out test set contains only **52 fraud cases** out of 42,722 transactions. A shift of just 2–3 caught frauds creates large swings in precision, recall, and PR-AUC.
- **Why it matters:** Senior interviewers will immediately spot that point estimates (e.g. comparing model A with PR-AUC 0.85 to model B with 0.84) are statistically indistinguishable without confidence intervals.
- **Current state:** Single-point metrics without confidence intervals or paired significance tests.

#### 2. [HIGH IMPACT] Code Duplication Between Notebooks and Scripts
- **Issue:** The three notebooks (`01_eda.ipynb`, `02_modeling.ipynb`, `03_cost_analysis.ipynb`) duplicate script logic inline rather than importing shared analysis utilities.
- **Why it matters:** Shows junior habit of copy-pasting code between research scripts and presentation notebooks, creating divergence risk when models or parameters change.

#### 3. [MEDIUM-HIGH IMPACT] Probability Distortion from `scale_pos_weight` in Cost Calculations
- **Issue:** XGBoost is trained with `scale_pos_weight ≈ 518`, shifting raw output probabilities away from true empirical risk. While rank order (and thus PR-AUC) is preserved, the cost-optimal per-transaction rule ($p \times \text{Amount} > \text{FP\_cost}$) requires calibrated probabilities.
- **Current state:** Notebook 02 explores isotonic and Platt calibration, but the main cost script (`run_cost.py`) relies on a global grid threshold search on raw scores rather than systematically evaluating calibrated probabilities across models.

#### 4. [MEDIUM IMPACT] Hard-Coded Parameters Across Files
- **Issue:** `SEED = 42` is in `src/fraud/data.py`, `FP = 5.0` is hard-coded in `run_cost.py` and `replay.py`, split fractions `(0.70, 0.15)` are scattered as defaults.
- **Why it matters:** Lacks a centralized configuration module (`config.py` or `config.yaml`), making scenario changes error-prone.

#### 5. [MEDIUM IMPACT] Lack of Single-Command Orchestration
- **Issue:** To reproduce the pipeline, a user must sequentially run `run_modeling.py`, `run_cost.py`, `run_explain.py`, `python -m fraud.replay`, and `pytest`.
- **Why it matters:** Friction for code reviewers. A single `python run_all.py` or `Makefile` is expected in professional repositories.

#### 6. [MEDIUM IMPACT] Static Horizon Evaluation
- **Issue:** Evaluating on a single static test window does not test model stability against temporal degradation or concept drift.
- **Why it matters:** In production fraud systems, models face rolling deployment. An expanding-window / rolling-origin evaluation demonstrates operational understanding.

---

### C. What is Missing (Ranked by Hiring Impact)

1. **Bootstrap Confidence Intervals:** No confidence intervals for PR-AUC or test costs (1,000 bootstrap resamples) to prove claims of model superiority.
2. **Rolling-Origin Evaluation:** No multi-period out-of-time evaluation to measure performance stability across sequential test slices.
3. **Feature Engineering & Ablation Table:** No domain-appropriate feature additions (e.g. `hour_of_day = (Time // 3600) % 24`, `log_amount = log1p(Amount)`) evaluated via systematic validation ablation.
4. **Analyst Review Capacity Constraint:** Real fraud operations have fixed human staffing. Missing an evaluation view of: "Given a budget of $N$ reviews/hour, what recall and loss avoidance does each model deliver?"
5. **Ensemble & Alternative Models:** No comparison with a competitive tree model (LightGBM/Random Forest) or rank-average ensemble of supervised + unsupervised anomaly signals.
6. **Centralized Pipeline Runner (`run_all.py`):** No single executable script to reproduce the entire pipeline from end to end.

---

## 3. Prioritized Fix List (Phase 2 to Phase 5 Proposal)

Below is the proposed sequence of improvements, structured to deliver maximum technical credibility:

| Phase | Priority | Focus | Deliverables |
|---|---|---|---|
| **Phase 2** | P0 | **Correctness & Infrastructure** | 1. Centralized `src/fraud/config.py` for seeds, paths, split fractions, and cost assumptions.<br>2. Single-command pipeline orchestrator `run_all.py`.<br>3. Extended unit tests (`test_data.py`, `test_features.py`, pipeline end-to-end sanity).<br>4. Clean notebook imports to eliminate script duplication. |
| **Phase 3** | P1 | **Rigorous Evaluation** | 1. 1,000-sample bootstrap confidence intervals for test PR-AUC and cost.<br>2. Paired bootstrap hypothesis test ($p$-value / difference CI) between XGBoost and Logistic Regression.<br>3. Rolling-origin evaluation across expanding temporal slices with spread metrics. |
| **Phase 4** | P1 | **Disciplined Feature & Model Expansion** | 1. Domain feature additions (`hour_of_day`, `log_amount`).<br>2. Validation-based ablation table reporting PR-AUC impact of each feature (drop non-performing ones).<br>3. Add LightGBM and Random Forest to benchmark suite.<br>4. Supervised + unsupervised rank-average ensemble (XGBoost + Isolation Forest). |
| **Phase 5** | P2 | **Decisioning, Calibration & Production Operations** | 1. Systematic calibration comparison (Raw vs Platt vs Isotonic) on validation set with Brier score.<br>2. Amount-aware decision rule ($p_{\text{cal}} \times \text{Amount} > \text{FP\_cost}$) benchmarked against single global threshold.<br>3. Analyst capacity constraints view (recall & cost under fixed review budgets: 5, 10, 20 reviews/hour).<br>4. Comprehensive README update pulling strictly from written result artifacts. |
