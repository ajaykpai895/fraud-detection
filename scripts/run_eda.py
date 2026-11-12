"""Exploratory data analysis script to generate and save figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from fraud.config import FIGURES_DIR
from fraud.data import load

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

df = load()
V = [f"V{i}" for i in range(1, 29)]

print("Dataset shape:", df.shape)
print("Class counts:\n", df["Class"].value_counts())
print(f"Fraud rate: {df['Class'].mean():.4%}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# 1. Amount by class
fig, ax = plt.subplots(figsize=(7, 4))
for c, lab, col in [(0, "legitimate", "#4878CF"), (1, "fraud", "#D65F5F")]:
    ax.hist(
        np.log10(df.loc[df.Class == c, "Amount"] + 1),
        bins=50,
        alpha=0.6,
        density=True,
        label=lab,
        color=col,
    )
ax.set_xlabel("log10(Amount + 1), USD")
ax.set_ylabel("density")
ax.set_title("Transaction amount by class")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_amount.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved eda_amount.png")

# 2. Fraud rate over time
df["hour"] = (df.Time // 3600).astype(int)
rate = df.groupby("hour")["Class"].agg(["mean", "sum", "size"])

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(rate.index, rate["mean"].mul(100), marker="o", ms=4, color="#4878CF")
ax.set_xlabel("hours since first transaction in dataset")
ax.set_ylabel("fraud rate (%)")
ax.set_title("Fraud rate over the two-day window")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_fraud_rate_time.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved eda_fraud_rate_time.png")

# 3. KS statistic separation
ks = pd.Series(
    {c: ks_2samp(df.loc[df.Class == 1, c], df.loc[df.Class == 0, c]).statistic for c in V}
).sort_values()

fig, ax = plt.subplots(figsize=(6, 8))
ks.plot.barh(ax=ax, color="#4878CF")
ax.set_xlabel("KS statistic (fraud vs legitimate)")
ax.set_title("Feature separation by class\n(higher = more separating)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_ks_features.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved eda_ks_features.png")

print("\nTop 5 separating features by KS:")
print(ks.tail(5))
