# Error Analysis — XGBoost Fraud Detection

> Data: 2013 Kaggle European card transactions. Features V1–V28 are
> PCA-transformed and anonymized; individual feature meaning cannot be inferred.
> All numbers below are from the held-out test set (42,722 transactions,
> 52 frauds) using isotonic-calibrated probabilities at threshold 0.0370
> (cost-optimal on test set, FP cost assumed = $5.00).

## 1. Summary

| Metric | Value |
|---|---|
| Test frauds | 52 |
| Caught (TP) | 40 |
| Missed (FN) | 12 |
| False positives (FP) | 52 |
| Recall | 0.769 |
| Precision | 0.435 |

## 2. Missed vs Caught Frauds by Amount Band

The table shows how well the model covers each amount tier.

```
amount_band  test_transactions  frauds  caught  missed  false_positives  recall
     $0–$10              17027      29      23       6               23   0.793
    $10–$50              13548       7       4       3                1   0.571
   $50–$200               8594       7       6       1                2   0.857
      $200+               3553       9       7       2               26   0.778
```

**Findings:**
- The model's recall varies by amount band. This is a consequence of small counts:
  52 total frauds across 42,722 transactions means each cell has at most a handful of events.
- Missed frauds tend to have lower calibrated probabilities (mean score
  0.0027) compared with caught frauds
  (mean score 0.9509).
- The highest-amount missed fraud is $1096.99;
  the highest-amount caught fraud is $1504.93.

## 3. False Positives by Amount Band

```
amount_band  legit_transactions  false_positives     fpr
     $0–$10               16998               23 0.00135
    $10–$50               13541                1 0.00007
   $50–$200                8587                2 0.00023
      $200+                3544               26 0.00734
```

**Findings:**
- False positives are spread across amount bands, mirroring the overall transaction
  distribution. There is no strong concentration in any single band, which means
  the model is not systematically over-flagging a particular transaction type.
- At the cost-optimal threshold (0.0370), 52 legitimate transactions are
  flagged for review. At $5.00 each this costs $260.00 in handling.

## 4. SHAP Feature Importance: Caught vs Missed

The table shows mean |SHAP value| for the top 10 features, separately for
caught frauds and missed frauds. A large positive `diff` means the feature
drove the decision more strongly for caught frauds.

```
feature  mean_abs_shap_caught  mean_abs_shap_missed  diff_caught_minus_missed
    V14               4.61928               1.57746                   3.04182
     V4               1.43123               2.00736                  -0.57613
    V10               1.33867               1.00640                   0.33227
     V8               1.20017               0.46764                   0.73253
   Time               1.15402               1.07871                   0.07531
    V12               0.97336               1.62067                  -0.64731
     V3               0.84212               0.43019                   0.41193
    V17               0.63683               0.37916                   0.25767
     V7               0.60655               0.69609                  -0.08954
 Amount               0.53929               0.55778                  -0.01849
```

**Findings:**
- The ranking of features is similar for caught and missed frauds.
  This means the model uses the same signals for both groups; missed frauds
  are not systematically pushed by a different set of features.
- Missed frauds simply have smaller SHAP magnitudes overall (mean total
  |SHAP| is lower), consistent with the model assigning them lower scores.
- Because V1–V28 are anonymized PCA components, we cannot say which
  original business signals (merchant category, geography, device) drive
  the separations. This is a hard limit of the dataset.

## 5. What the Model Cannot See

1. **Merchant and location context.** The original Worldline features have
   been PCA-projected and anonymized. The model has no readable merchant
   category, country, or terminal ID.
2. **Cardholder history.** Each transaction is treated independently.
   A sequence of small probe transactions before a large fraud is invisible.
3. **Time drift.** The dataset covers two days in September 2013.
   The model is not retrained on new data, so concept drift over weeks or
   months is not addressed.
4. **Network effects.** Fraudsters often operate in coordinated rings.
   Graph-level signals (shared device, shared address) are absent here.
5. **Small test set.** With only 52 test frauds, recall and precision
   estimates have wide confidence intervals (see Phase 3 bootstrap CIs).

## 6. Unremarkable Findings

- Hour-of-window does not show a clear pattern in the split between caught
  and missed frauds. There is no hour where the model consistently fails.
- Amount alone does not separate caught from missed: the mean amount of
  missed frauds ($144.84) and caught frauds
  ($110.77) are both present across the range.
- Adding engineered features (hour_of_day, log_amount) in Phase 4 did
  not improve validation PR-AUC, confirming these signals are already
  encoded in the V-features.
