# Cost-Sensitive Credit Card Fraud Detection

An end-to-end fraud decisioning system built on real payment transactions, framed around business loss tradeoffs rather than raw classification accuracy.

---

## 1. Problem Framing & Cost Decisioning

Fraud detection is an asymmetric loss problem:
- **False Negative (FN):** A missed fraud causes direct financial loss equal to the full transaction amount.
- **False Positive (FP):** An erroneous alert incurs operational review expense and customer friction, assumed at a fixed \$5.00 handling cost.
- **True Positive / True Negative:** Assumed \$0.00 net penalty.

At 0.1727% positive prevalence (492 frauds in 284,807 transactions), accuracy and ROC-AUC are misleading. A model flagging zero transactions achieves 99.83% accuracy but loses \$6,168.88 on test. We evaluate all models primarily by **PR-AUC (Average Precision)** and **Total Decision Cost (USD)**.

---

## 2. Validation Design

Transactions are deployed forward in time. To avoid temporal data leakage, we enforce:
1. **Time-Based Chronological Split:**
   - Train (70%): First 199,364 transactions (384 frauds)
   - Validation (15%): Next 42,721 transactions (56 frauds)
   - Test (15%): Final 42,722 transactions (52 frauds; no-skill PR-AUC = 0.00122)
2. **Leakage-Free Preprocessing:** RobustScaler centers on the median and IQR of the training set only. PCA features (V1–V28) pass through unscaled.
3. **Threshold Selection on Validation Only:** Decision thresholds are selected on the validation set to minimize expected loss, then evaluated once on the unseen test set.

---

## 3. Results & Uncertainty Quantification

All metrics below are computed from code on the held-out future test set. Confidence intervals reflect 1,000 percentile bootstrap resamples.

### Model Comparison

| Model | Val PR-AUC | Test PR-AUC | 95% Bootstrap CI | Test Recall@P80 | Test ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | 0.8713 | **0.7780** | — | 0.7500 | 0.9460 |
| **XGBoost (Tuned)** | 0.8670 | 0.7717 | [0.6541, 0.8785] | 0.7500 | 0.9696 |
| **XGBoost + SMOTE** | 0.8444 | 0.7746 | [0.6586, 0.8835] | 0.7500 | 0.9733 |
| **Logistic Regression** | 0.8394 | 0.7066 | [0.5674, 0.8508] | 0.7308 | 0.9773 |
| **LightGBM** | 0.6612 | 0.7068 | — | 0.6923 | 0.8701 |
| **Isolation Forest** | 0.0449 | 0.0640 | [0.0403, 0.1028] | 0.0000 | 0.9442 |
| *No-Skill Baseline* | 0.0013 | 0.0012 | [0.0009, 0.0016] | — | 0.5000 |

### Statistical Significance (Paired Bootstrap, 1,000 Draws)
- **XGBoost vs. Logistic Regression:** $\Delta \text{PR-AUC} = +0.0651$, 95% CI: [−0.0129, +0.1570], $p = 0.1940$ (not statistically significant at $\alpha = 0.05$). The direction favors XGBoost, but with only 52 test frauds, the confidence interval spans zero.
- **XGBoost vs. XGBoost + SMOTE:** $\Delta \text{PR-AUC} = -0.0028$, 95% CI: [−0.0275, +0.0176], $p = 0.8420$. SMOTE adds no statistically detectable improvement over class weighting.

### Financial Loss & Decision Thresholds

| Strategy | Threshold | Alerts | Caught | Missed | Test Cost | Loss Avoided |
|---|---|---|---|---|---|---|
| Flag Nothing Baseline | — | 0 | 0 | 52 | \$6,168.88 | 0.0% |
| Default Probability (0.50) | 0.5000 | 45 | 39 | 13 | \$2,407.40 | 61.0% |
| **Cost-Optimal Threshold** | **0.0082** | **117** | **40** | **12** | **\$2,123.10** | **65.6%** |
| Isotonic Cost-Optimal | 0.0370 | 92 | 40 | 12 | \$1,998.10 | 67.6% |
| Amount-Aware Rule ($p \times \text{Amt} > \$5$) | Dynamic | 108 | 19 | 33 | \$1,856.68 | 69.9% |

The cost-optimal threshold (\$2,123.10, 95% CI: [\$469.89, \$4,874.38]) saves \$4,045.78 net over flagging nothing, avoiding 65.6% of gross losses.

---

## 4. Key Findings

1. **Threshold tuning yields direct dollar gains:** Lowering the operational threshold from 0.50 to 0.0082 avoids an additional \$284.30 in test losses by catching high-value frauds that conventional thresholds ignore.
2. **Unsupervised isolation performs poorly alone:** Isolation Forest achieves a test PR-AUC of 0.0640, proving that general transaction outliers rarely coincide with actual fraud patterns.
3. **Feature engineering ablation:** Adding `hour_of_day` and `log_amount` decreased validation PR-AUC by 0.0143 (0.8527 vs. 0.8670). The derived columns proved redundant because the PCA components already encode non-linear combinations of time and spend.
4. **Probability calibration:** Isotonic calibration reduces the test Brier score from 0.000438 to 0.000399 (a 9% improvement).
5. **Operational capacity:** At a realistic constraint of 20 analyst alerts/hour, the model catches 40 of 52 frauds (76.9% recall) and avoids 65.3% of losses (\$2,138.10 cost).

---

## 5. Stated Assumptions & Limitations

- **Historical Snapshot:** The dataset consists of 48 hours of European credit card transactions from September 2013, not a real-time stream. It does not reflect concept drift or modern payment rails (e.g. contactless, Apple Pay).
- **Anonymized PCA Features:** Features V1–V28 are PCA projections. While V14 and V4 have the highest SHAP impact, their underlying business identities (e.g. MCC, terminal type, merchant country) cannot be determined.
- **Assumed Operational Costs:** The \$5.00 false-positive handling cost and \$0.00 review cost for caught frauds are stated baseline assumptions. Sensitivity analysis across \$1–\$50 shows the model maintains cost superiority across all tiers.
- **Sample Size Constraint:** The test set contains only 52 fraud events, resulting in wide confidence intervals on test recall and precision.

---

## 6. Reproduction

```bash
# Clone and install in editable mode
pip install -e .

# Run the complete end-to-end pipeline (reproduces all tables and figures)
python run_all.py

# Run test suite (27 unit tests)
pytest

# Launch analyst review dashboard
streamlit run app.py
```
