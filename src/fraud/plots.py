"""Shared visualization styles and plotting utilities for portfolio figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Consistent styling constants
NAVY = "#2B4C7E"
CORAL = "#D65F5F"
GREEN = "#2E7D32"
PURPLE = "#7B1FA2"
SLATE = "#64748B"
LIGHT_GREY = "#E2E8F0"

MODEL_COLORS = {
    "logreg": NAVY,
    "iforest": GREEN,
    "xgb": CORAL,
    "xgb_smote": PURPLE,
    "rf": "#F59E0B",
    "lgbm": "#0284C7",
    "ensemble_xgb_lgbm": SLATE,
    "no_skill": "#94A3B8",
}


def apply_style():
    """Apply standard clean styling across all Matplotlib plots."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "font.family": "sans-serif",
        }
    )
