"""
=============================================================
  Project 2 — Customer Churn Prediction
  Script 1: Exploratory Data Analysis (EDA)
=============================================================
Run:
    python churn_eda.py
Output:
    eda_plots/  — folder containing all saved PNG charts
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
COLORS = {"Yes": "#E74C3C", "No": "#2ECC71"}
OUT_DIR = "eda_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────
print("=" * 55)
print("  CUSTOMER CHURN — EXPLORATORY DATA ANALYSIS")
print("=" * 55)

df_raw = pd.read_csv("churn_prediction.csv")
print(f"\nRaw shape : {df_raw.shape}")

df = df_raw.copy()

# --- Fix inconsistent categorical labels ---
df["Gender"] = (
    df["Gender"]
    .str.strip()
    .replace({"M": "Male", "F": "Female"})
)
df["IsActive"] = (
    df["IsActive"]
    .str.strip()
    .replace({"Y": "Yes", "N": "No"})
)
df["Country"] = (
    df["Country"]
    .str.strip()
    .replace({"IND": "India", "IN": "India"})
)

# --- Handle missing values ---
cat_cols  = ["Gender", "ProductCategory", "PaymentMethod",
             "IsActive", "Browser", "Device", "Country"]
num_cols  = ["Returns", "ReviewScore"]

for c in cat_cols:
    df[c].fillna(df[c].mode()[0], inplace=True)

for c in num_cols:
    df[c].fillna(df[c].median(), inplace=True)

# DiscountUsed — mixed type; coerce to numeric then fill median
df["DiscountUsed"] = pd.to_numeric(df["DiscountUsed"], errors="coerce")
df["DiscountUsed"].fillna(df["DiscountUsed"].median(), inplace=True)

# Drop columns with too much noise / leakage
df.drop(columns=["CustomerID", "City", "State", "LastPurchaseDate"], inplace=True)

print(f"Clean shape: {df.shape}")
print(f"Missing after cleaning: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────
# 2. BASIC STATS
# ─────────────────────────────────────────────────────────
print("\n── Churn Distribution ──")
churn_counts = df["Churn"].value_counts()
churn_pct    = df["Churn"].value_counts(normalize=True) * 100
print(pd.DataFrame({"Count": churn_counts, "Pct (%)": churn_pct.round(1)}))

print("\n── Numeric Summary ──")
print(df.describe(include=np.number).round(2))

# ─────────────────────────────────────────────────────────
# 3. PLOT HELPERS
# ─────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────────────────
# 4. PLOT 1 — Churn Distribution (pie + bar)
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Churn Distribution", fontsize=15, fontweight="bold")

# Pie
axes[0].pie(
    churn_counts,
    labels=churn_counts.index,
    autopct="%1.1f%%",
    colors=[COLORS[l] for l in churn_counts.index],
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
axes[0].set_title("Churn Share")

# Bar
bars = axes[1].bar(churn_counts.index, churn_counts.values,
                   color=[COLORS[l] for l in churn_counts.index],
                   edgecolor="white", linewidth=1.5, width=0.5)
for bar, val in zip(bars, churn_counts.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 20, str(val),
                 ha="center", va="bottom", fontweight="bold")
axes[1].set_title("Churn Count")
axes[1].set_ylabel("Number of Customers")
axes[1].set_xlabel("Churn")
save(fig, "01_churn_distribution.png")

# ─────────────────────────────────────────────────────────
# 5. PLOT 2 — Numeric feature distributions by Churn
# ─────────────────────────────────────────────────────────
num_features = ["Age", "Income", "SpendingScore",
                "PurchaseAmount", "ReviewScore",
                "SessionTime", "Returns", "DiscountUsed"]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Numeric Feature Distributions by Churn", fontsize=15, fontweight="bold")
axes = axes.flatten()

for ax, feat in zip(axes, num_features):
    for label, grp in df.groupby("Churn"):
        sns.kdeplot(grp[feat], ax=ax, label=label,
                    color=COLORS[label], fill=True, alpha=0.35, linewidth=2)
    ax.set_title(feat)
    ax.set_xlabel("")
    ax.legend(title="Churn")

plt.tight_layout()
save(fig, "02_numeric_distributions.png")

# ─────────────────────────────────────────────────────────
# 6. PLOT 3 — Categorical features vs Churn
# ─────────────────────────────────────────────────────────
cat_features = ["Gender", "ProductCategory", "PaymentMethod",
                "IsActive", "Browser", "Device"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Categorical Features vs Churn", fontsize=15, fontweight="bold")
axes = axes.flatten()

for ax, feat in zip(axes, cat_features):
    ct = (df.groupby([feat, "Churn"])
            .size()
            .reset_index(name="Count"))
    pivot = ct.pivot(index=feat, columns="Churn", values="Count").fillna(0)
    pivot.plot(kind="bar", ax=ax,
               color=[COLORS[c] for c in pivot.columns],
               edgecolor="white", linewidth=1)
    ax.set_title(feat)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Churn")

plt.tight_layout()
save(fig, "03_categorical_vs_churn.png")

# ─────────────────────────────────────────────────────────
# 7. PLOT 4 — Churn Rate by Category
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Churn Rate (%) by Category", fontsize=15, fontweight="bold")
axes = axes.flatten()

for ax, feat in zip(axes, cat_features):
    rate = (df.groupby(feat)["Churn"]
              .apply(lambda x: (x == "Yes").mean() * 100)
              .sort_values(ascending=False)
              .reset_index())
    rate.columns = [feat, "ChurnRate"]
    bars = ax.barh(rate[feat], rate["ChurnRate"],
                   color="#E74C3C", edgecolor="white", linewidth=1)
    for bar, val in zip(bars, rate["ChurnRate"]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_title(f"Churn Rate by {feat}")
    ax.set_xlabel("Churn Rate (%)")
    ax.set_xlim(0, 100)

plt.tight_layout()
save(fig, "04_churn_rate_by_category.png")

# ─────────────────────────────────────────────────────────
# 8. PLOT 5 — Correlation Heatmap
# ─────────────────────────────────────────────────────────
df_enc = df.copy()
df_enc["Churn_bin"] = (df_enc["Churn"] == "Yes").astype(int)
corr = df_enc[num_features + ["Churn_bin"]].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, linewidths=0.5,
            ax=ax, square=True)
ax.set_title("Correlation Heatmap (Numeric Features + Churn)",
             fontsize=13, fontweight="bold")
save(fig, "05_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────
# 9. PLOT 6 — Box plots of key numerics by Churn
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Box Plots — Numeric Features by Churn", fontsize=15, fontweight="bold")
axes = axes.flatten()

for ax, feat in zip(axes, num_features):
    sns.boxplot(data=df, x="Churn", y=feat, ax=ax,
                palette=COLORS, linewidth=1.5)
    ax.set_title(feat)
    ax.set_xlabel("Churn")

plt.tight_layout()
save(fig, "06_boxplots.png")

# ─────────────────────────────────────────────────────────
# 10. PLOT 7 — Age & Income bins vs Churn Rate
# ─────────────────────────────────────────────────────────
df["AgeGroup"]    = pd.cut(df["Age"],    bins=[18, 30, 40, 50, 60, 70],
                            labels=["18-30","30-40","40-50","50-60","60-70"])
df["IncomeGroup"] = pd.cut(df["Income"], bins=5,
                            labels=["Very Low","Low","Medium","High","Very High"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Churn Rate by Age Group & Income Group", fontsize=13, fontweight="bold")

for ax, feat in zip(axes, ["AgeGroup", "IncomeGroup"]):
    rate = (df.groupby(feat, observed=True)["Churn"]
              .apply(lambda x: (x == "Yes").mean() * 100))
    rate.plot(kind="bar", ax=ax, color="#E74C3C", edgecolor="white",
              linewidth=1.5, width=0.6)
    ax.set_title(f"Churn Rate by {feat}")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    for i, val in enumerate(rate):
        ax.text(i, val + 0.5, f"{val:.1f}%", ha="center", fontsize=9)

plt.tight_layout()
save(fig, "07_age_income_churnrate.png")

# ─────────────────────────────────────────────────────────
# 11. SUMMARY
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  EDA COMPLETE — Saved 7 plots to:", OUT_DIR)
print("=" * 55)
print("\nKey Takeaways:")
print(f"  • Dataset    : {df.shape[0]:,} customers, {df.shape[1]} features")
print(f"  • Churn rate : {(df['Churn']=='Yes').mean()*100:.1f}% (nearly balanced)")
print(f"  • Top correlations with Churn: see heatmap (plot 05)")
print(f"  • Run churn_ml.py next to train models")
