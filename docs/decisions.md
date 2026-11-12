# Architecture & Modeling Decisions

This document records the core technical and methodology decisions made in this project, explaining why each choice was adopted over standard alternatives.

---

## 1. Why Time-Based Split Instead of Random Split

In credit card fraud detection, models are trained on past data and deployed to score incoming future transactions. Evaluating models with random k-fold cross-validation or random train/test splits introduces severe temporal leakage:
- Fraudsters frequently execute rapid bursts of coordinated transactions. Random splitting places transactions from the same attack burst into both training and test sets, artificially inflating recall.
- Cardholder transaction frequency, amount distributions, and merchant patterns drift over time.
- A random split evaluates interpolation within a known time period; production deployment requires forward extrapolation in time.

**Our Choice:** We strictly ordered all 284,807 transactions by their `Time` field and applied a forward chronological split:
- Training set: First 70% (199,364 transactions, 384 frauds)
- Validation set: Next 15% (42,721 transactions, 56 frauds)
- Test set: Final 15% (42,722 transactions, 52 frauds)

All scaler parameters (median and IQR from `RobustScaler`), threshold optimization, and probability calibrators are fitted exclusively on the training or validation windows. The test set is scored once at the very end.

---

## 2. Why PR-AUC Over ROC-AUC

With an extreme class imbalance of ~0.1727% (1 fraud per 580 legitimate transactions), ROC-AUC is misleading:
- The false positive rate is defined as $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$.
- In our test set, $\text{TN} \approx 42,670$. If the model generates 200 false alarms, $\text{FPR} \approx \frac{200}{42,870} \approx 0.0047$ (less than 0.5%).
- A model with 200 false alarms and 40 true positives has an ROC curve that looks almost perfect ($\text{ROC-AUC} > 0.97$), yet its precision is only $40 / 240 = 16.7\%$—meaning 5 out of 6 analyst alerts are false alarms.
- **PR-AUC (Average Precision)** focuses strictly on minority class performance: Precision vs. Recall. No-skill PR-AUC equals class prevalence (0.00122 on test), providing a true signal-to-noise benchmark.

---

## 3. Why $5.00 False Positive Cost Was Chosen

Fraud detection is a cost-asymmetric business problem, not an accuracy competition:
- A False Negative (missed fraud) costs the financial institution the transaction amount (direct loss).
- A False Positive (false alarm) incurs operational review expense and customer friction:
  - Automated step-up authentication (SMS / in-app OTP) has carrier/infra costs and causes transaction drop-off.
  - Manual review by a tier-1 fraud analyst costs estimated analyst time (\$15–\$25/hr wage $\times$ 2–3 minutes review = \$1–\$3).
  - Customer friction and potential card re-issuance add indirect costs.
- Industry operational benchmarks place total false alarm handling and friction cost between \$2.00 and \$10.00. We adopted \$5.00 as a transparent, defensible central assumption.
- To avoid dependence on this single number, we ran a sensitivity sweep across \$1, \$2, \$5, \$10, \$20, and \$50, showing that the model's cost savings remain superior to the baseline across all realistic values.

---

## 4. Why SMOTE Was Evaluated and Whether It Helped

Oversampling with SMOTE (Synthetic Minority Over-sampling Technique) is widely cited in academic literature. We evaluated whether SMOTE improves performance when tree models already support class weighting (`scale_pos_weight`):
- **Empirical result on test set:**
  - Tuned XGBoost: PR-AUC = 0.7717 (95% CI: [0.6541, 0.8785])
  - XGBoost + SMOTE: PR-AUC = 0.7746 (95% CI: [0.6586, 0.8835])
- **Hypothesis test:** Paired bootstrap difference $\Delta = -0.0028$, $p = 0.8420$. The difference is statistically indistinguishable from zero.
- **Conclusion:** SMOTE creates synthetic points by linear interpolation between nearest minority neighbors in 30-dimensional space. Because V1–V28 are PCA projections with complex manifold geometries, synthetic points often fall in low-density or ambiguous regions. Tree algorithms handle class weighting natively with lower compute overhead and zero synthetic artifact risk.

---

## 5. What We Would Do Next With Live Data

If this system were extended to an active payment processing stream, the primary engineering additions would be:
1. **Streaming Velocity Features:** Rolling transaction counts and aggregate spend per cardholder over 5-minute, 1-hour, 24-hour, and 7-day sliding windows (e.g. using Redis or Apache Flink).
2. **Delayed Feedback Architecture:** Real-world chargebacks take 30 to 90 days to settle. Models must be trained with observation delays and label-delay estimation to avoid false non-fraud assumptions.
3. **Graph Risk Signals:** Shared device identifiers, IP subnets, billing address clusters, and merchant risk scores to detect coordinated fraud rings.
4. **Adaptive Thresholding:** Dynamic threshold adjustment based on real-time analyst queue capacity (e.g. tightening thresholds during peak shopping hours).
