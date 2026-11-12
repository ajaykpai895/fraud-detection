"""Streamlit Decisioning Review Tool for Fraud Analysts.

Evaluates transactions from the held-out test set with calibrated fraud probabilities,
cost-sensitive decisions, and local SHAP explanations.
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from fraud.calibration import apply_calibrator, fit_isotonic
from fraud.config import DEFAULT_FP_COST, MODELS_DIR, RESULTS_DIR
from fraud.data import load
from fraud.features import FEATS_BASE, make_pre
from fraud.split import time_split

st.set_page_config(
    page_title="Fraud Decisioning Review Tool",
    page_icon=None,
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    """Load test dataset, fitted model, and isotonic calibrator."""
    df = load()
    tr, va, te = time_split(df)

    model_path = MODELS_DIR / "xgb.joblib"
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}. Please run pipeline scripts first.")
        st.stop()

    model = joblib.load(model_path)
    if hasattr(model, "named_steps"):
        clf = model.named_steps["clf"]
        pre = model.named_steps["pre"]
    else:
        clf = model
        pre = make_pre(FEATS_BASE).fit(tr[FEATS_BASE])

    # Fit isotonic calibrator using validation predictions
    Xva_t = pre.transform(va[FEATS_BASE])
    s_va = clf.predict_proba(Xva_t)[:, 1]
    y_va = va["Class"].values
    calibrator = fit_isotonic(y_va, s_va)

    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(clf)

    # Default optimal threshold
    chosen_thr_path = RESULTS_DIR / "chosen_threshold.json"
    default_thr = 0.0370
    if chosen_thr_path.exists():
        try:
            with open(chosen_thr_path) as f:
                cfg = json.load(f)
                default_thr = float(cfg.get("threshold", 0.0370))
        except Exception:
            pass

    return te.reset_index(drop=True), clf, pre, calibrator, explainer, default_thr


te_df, clf, pre, calibrator, explainer, default_thr = load_artifacts()

# Top Banner / Header
st.title("Fraud Decisioning Review Tool")
st.caption(
    "Evaluation snapshot from September 2013 European card transactions. "
    "Features V1–V28 are PCA-transformed and anonymized. "
    "Framed around asymmetric financial cost: False Negative = Transaction Amount, False Positive = Review Cost."
)

st.markdown("---")

# Sidebar Controls
st.sidebar.subheader("Review Parameters")

# Preset selector for quick analyst inspection
fraud_indices = te_df[te_df["Class"] == 1].index.tolist()
legit_indices = te_df[te_df["Class"] == 0].index.tolist()

preset_options = {
    f"Fraud Sample 1 (${te_df.loc[fraud_indices[0], 'Amount']:.2f})": fraud_indices[0],
    f"Fraud Sample 2 (${te_df.loc[fraud_indices[1], 'Amount']:.2f})": fraud_indices[1],
    f"Fraud Sample 3 (${te_df.loc[fraud_indices[2], 'Amount']:.2f})": fraud_indices[2],
    f"Legitimate Sample 1 (${te_df.loc[legit_indices[0], 'Amount']:.2f})": legit_indices[0],
    f"Legitimate Sample 2 (${te_df.loc[legit_indices[100], 'Amount']:.2f})": legit_indices[100],
    "Custom Transaction Index": -1,
}

selection = st.sidebar.selectbox("Select Transaction Scenario", list(preset_options.keys()))

if preset_options[selection] == -1:
    selected_idx = int(
        st.sidebar.number_input(
            f"Test Transaction Index (0 to {len(te_df) - 1})",
            min_value=0,
            max_value=len(te_df) - 1,
            value=0,
            step=1,
        )
    )
else:
    selected_idx = preset_options[selection]

# Threshold & Cost Configuration
threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.001,
    max_value=0.999,
    value=float(default_thr),
    step=0.001,
    format="%.4f",
    help="Transactions with calibrated probability at or above this threshold are flagged for manual review.",
)

fp_cost = st.sidebar.number_input(
    "Assumed FP Review Cost ($)",
    min_value=0.5,
    max_value=100.0,
    value=float(DEFAULT_FP_COST),
    step=0.5,
    help="Fixed operational handling and customer friction cost per false alarm.",
)

# Extract Row and Predict
row = te_df.iloc[selected_idx]
X_row = pd.DataFrame([row[FEATS_BASE]])
X_t = pre.transform(X_row)

raw_score = float(clf.predict_proba(X_t)[0, 1])
cal_prob = float(apply_calibrator(calibrator, [raw_score])[0])

is_flagged = cal_prob >= threshold
amount = float(row["Amount"])
true_class = int(row["Class"])
expected_loss = cal_prob * amount
aware_flagged = expected_loss > fp_cost

# Main KPI Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Transaction Amount", f"${amount:.2f}")

with col2:
    st.metric("Calibrated Probability", f"{cal_prob:.4f}", f"Raw score: {raw_score:.4f}")

with col3:
    status_label = "FLAG FOR REVIEW" if is_flagged else "APPROVE"
    status_color = "inverse" if is_flagged else "normal"
    st.metric("Model Decision", status_label, f"Threshold: {threshold:.4f}")

with col4:
    ground_truth = "Fraud (Class 1)" if true_class == 1 else "Legitimate (Class 0)"
    st.metric("True Historical Label", ground_truth)

st.markdown("---")

# Decision Logic & Cost Impact
c_left, c_right = st.columns([1, 1])

with c_left:
    st.subheader("Decision Breakdown")
    comparison_data = {
        "Rule": ["Global Cost-Optimal Threshold", "Amount-Aware Rule (p × Amount > FP Cost)"],
        "Condition": [f"p >= {threshold:.4f}", f"p × ${amount:.2f} > ${fp_cost:.2f}"],
        "Evaluation": [
            f"{cal_prob:.4f} >= {threshold:.4f}",
            f"${expected_loss:.2f} > ${fp_cost:.2f}",
        ],
        "Action": [
            "FLAG" if is_flagged else "APPROVE",
            "FLAG" if aware_flagged else "APPROVE",
        ],
    }
    st.table(pd.DataFrame(comparison_data))

    # Operational financial implication
    if is_flagged and true_class == 1:
        st.success(
            f"True Positive: Fraud caught. Net saving = ${amount - fp_cost:.2f} (Amount ${amount:.2f} saved less ${fp_cost:.2f} review cost)."
        )
    elif is_flagged and true_class == 0:
        st.warning(
            f"False Positive: Legitimate transaction flagged. Incurred review handling cost of ${fp_cost:.2f}."
        )
    elif not is_flagged and true_class == 1:
        st.error(f"False Negative: Missed fraud. Direct financial loss of ${amount:.2f}.")
    else:
        st.info("True Negative: Legitimate transaction approved smoothly. Cost = $0.00.")

with c_right:
    st.subheader("Feature Values")
    # Show key feature values in a clean table
    feats_subset = ["Time", "Amount", "V14", "V4", "V10", "V12", "V17", "V8"]
    feat_display = pd.DataFrame(
        {
            "Feature": feats_subset,
            "Value": [round(float(row[f]), 4) for f in feats_subset],
        }
    )
    st.dataframe(feat_display, width="stretch", hide_index=True)

st.markdown("---")

# Explainability: SHAP Contribution
st.subheader("Local SHAP Attribution (Top Risk Contributors)")
st.caption(
    "Shows how individual anonymized PCA features shift the log-odds toward fraud (positive) or legitimate (negative)."
)

shap_val = explainer(X_t)[0]
scaled_names = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
feature_shap_pairs = list(zip(scaled_names, shap_val.values))
feature_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

top_k = feature_shap_pairs[:10]
top_names = [x[0] for x in top_k][::-1]
top_values = [x[1] for x in top_k][::-1]
bar_colors = ["#D65F5F" if v > 0 else "#4878CF" for v in top_values]

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.barh(top_names, top_values, color=bar_colors, height=0.6)
ax.axvline(0, color="#94A3B8", linewidth=0.8, linestyle="--")
ax.set_xlabel("SHAP Value (Impact on Model Log-Odds)", fontsize=9)
ax.set_title("Top 10 Feature Contributions for Selected Transaction", fontsize=10, loc="left")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)
