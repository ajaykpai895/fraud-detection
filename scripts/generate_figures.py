"""Generate all polished portfolio figures with consistent styling and units."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from fraud.config import FIGURES_DIR, RESULTS_DIR
from fraud.plots import MODEL_COLORS, NAVY, SLATE, apply_style

apply_style()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. PR Curves on Held-Out Test Set
# -------------------------------------------------------------
scores_path = RESULTS_DIR / "scores.npz"
if scores_path.exists():
    d = np.load(scores_path, allow_pickle=False)
    yte = d["yte"]
    no_skill = float(yte.mean())

    fig, ax = plt.subplots(figsize=(7, 4.5))

    plot_models = [
        ("xgb", "XGBoost (Tuned)"),
        ("xgb_smote", "XGBoost + SMOTE"),
        ("rf", "Random Forest"),
        ("logreg", "Logistic Regression"),
        ("lgbm", "LightGBM"),
        ("iforest", "Isolation Forest"),
    ]

    for key, label in plot_models:
        score_key = f"{key}_te"
        if score_key in d:
            s = d[score_key]
            p, r, _ = precision_recall_curve(yte, s)
            color = MODEL_COLORS.get(key, SLATE)
            lw = 1.6 if key in ["xgb", "rf"] else 1.1
            ax.plot(r, p, label=label, color=color, linewidth=lw)

    ax.axhline(
        no_skill,
        linestyle="--",
        color="#94A3B8",
        linewidth=1.0,
        label=f"No Skill Baseline ({no_skill:.4f})",
    )
    ax.set_xlabel("Recall (Fraction of Frauds Caught)")
    ax.set_ylabel("Precision (Fraction of Alerts that are Fraud)")
    ax.set_title("Precision-Recall Curves — Held-Out Test Set (42,722 rows, 52 frauds)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Generated reports/figures/pr_curves.png")

# -------------------------------------------------------------
# 2. Model Comparison with 95% Bootstrap Confidence Intervals
# -------------------------------------------------------------
ci_path = RESULTS_DIR / "model_comparison_with_ci.csv"
if ci_path.exists():
    ci_df = pd.read_csv(ci_path)
    # Sort descending by test PR-AUC
    ci_df = ci_df.sort_values("test_pr_auc", ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    y_pos = np.arange(len(ci_df))

    pts = ci_df["test_pr_auc"].values
    err_low = pts - ci_df["ci_95_lower"].values
    err_high = ci_df["ci_95_upper"].values - pts

    model_names = {
        "xgb_smote": "XGBoost + SMOTE",
        "xgb": "XGBoost (Tuned)",
        "logreg": "Logistic Regression",
        "iforest": "Isolation Forest",
    }
    labels = [model_names.get(m, m) for m in ci_df["model"]]
    colors = [MODEL_COLORS.get(m, NAVY) for m in ci_df["model"]]

    ax.errorbar(
        pts,
        y_pos,
        xerr=[err_low, err_high],
        fmt="o",
        color=NAVY,
        ecolor=SLATE,
        elinewidth=1.5,
        capsize=4,
        markersize=6,
    )

    for i, (pt, low, high) in enumerate(zip(pts, ci_df["ci_95_lower"], ci_df["ci_95_upper"])):
        ax.text(
            pt,
            i + 0.18,
            f"{pt:.4f} [{low:.4f}, {high:.4f}]",
            ha="center",
            fontsize=8,
            color="#334155",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test PR-AUC (Average Precision)")
    ax.set_title("Model Performance with 95% Bootstrap Confidence Intervals (1,000 Draws)")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison_ci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Generated reports/figures/model_comparison_ci.png")

# -------------------------------------------------------------
# 3. Cost Sensitivity Across FP Review Cost ($1 to $50)
# -------------------------------------------------------------
sens_path = RESULTS_DIR / "cost_sensitivity.csv"
if sens_path.exists():
    sens_df = pd.read_csv(sens_path)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    for m in ["xgb", "xgb_smote", "logreg", "iforest"]:
        sub = sens_df[sens_df["model"] == m].sort_values("fp_cost")
        color = MODEL_COLORS.get(m, SLATE)
        label_map = {
            "xgb": "XGBoost",
            "xgb_smote": "XGBoost + SMOTE",
            "logreg": "Logistic Regression",
            "iforest": "Isolation Forest",
        }
        ax.plot(
            sub["fp_cost"],
            sub["cost"],
            marker="o",
            label=label_map.get(m, m),
            color=color,
            linewidth=1.5,
            markersize=4,
        )

    flag_nothing = 6168.88  # constant flag-nothing loss on test
    ax.axhline(
        flag_nothing,
        linestyle=":",
        color="#DC2626",
        linewidth=1.2,
        label=f"Flag-Nothing Baseline (${flag_nothing:,.0f})",
    )

    ax.set_xlabel("Assumed False Positive Review Cost (USD)")
    ax.set_ylabel("Total Financial Loss on Test Set (USD)")
    ax.set_title("Sensitivity Analysis: Total Decision Cost vs FP Review Cost ($1 – $50)")
    ax.grid(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cost_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Generated reports/figures/cost_sensitivity.png")
