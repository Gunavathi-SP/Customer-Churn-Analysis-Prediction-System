"""
=============================================================
  Project 2 — Customer Churn Prediction
  Script 2: Machine Learning — Train, Evaluate & Save Models
=============================================================
Run:
    python churn_ml.py
Output:
    ml_plots/   — evaluation charts
    models/     — saved model + preprocessor (.pkl files)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.preprocessing     import OneHotEncoder
from sklearn.impute            import SimpleImputer
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics           import (accuracy_score, precision_score, recall_score,
                                       f1_score, roc_auc_score, roc_curve,
                                       confusion_matrix, classification_report)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

OUT_DIR   = "ml_plots"
MODEL_DIR = "models"
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────
print("=" * 55)
print("  CUSTOMER CHURN — ML PIPELINE")
print("=" * 55)

df = pd.read_csv("churn_prediction.csv")

# Fix inconsistent labels (same as EDA)
df["Gender"]  = df["Gender"].str.strip().replace({"M": "Male",  "F": "Female"})
df["IsActive"]= df["IsActive"].str.strip().replace({"Y": "Yes", "N": "No"})
df["Country"] = df["Country"].str.strip().replace({"IND": "India", "IN": "India"})

# Drop leaky / irrelevant columns
df.drop(columns=["CustomerID", "City", "State", "LastPurchaseDate"], inplace=True)

# DiscountUsed — coerce to numeric
df["DiscountUsed"] = pd.to_numeric(df["DiscountUsed"], errors="coerce")

# Target
df["Churn"] = (df["Churn"] == "Yes").astype(int)

X = df.drop(columns=["Churn"])
y = df["Churn"]

# ─────────────────────────────────────────────────────────
# 2. DEFINE FEATURE COLUMNS
# ─────────────────────────────────────────────────────────
num_features = ["Age", "Income", "SpendingScore", "PurchaseAmount",
                "Returns", "DiscountUsed", "ReviewScore", "SessionTime"]

cat_features = ["Gender", "ProductCategory", "PaymentMethod",
                "IsActive", "Browser", "Device", "Country"]

# ─────────────────────────────────────────────────────────
# 3. BUILD PREPROCESSOR
# ─────────────────────────────────────────────────────────
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])

cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_features),
    ("cat", cat_pipe, cat_features),
])

# ─────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ─────────────────────────────────────────────────────────
# 5. DEFINE MODELS
# ─────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=200, max_depth=10,
                                                   random_state=42, n_jobs=-1),
    "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                        max_depth=4, random_state=42),
}

# ─────────────────────────────────────────────────────────
# 6. TRAIN, EVALUATE & COLLECT METRICS
# ─────────────────────────────────────────────────────────
results   = {}
pipelines = {}

for name, clf in models.items():
    print(f"\n── Training: {name} ──")
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall"   : recall_score(y_test, y_pred),
        "F1"       : f1_score(y_test, y_pred),
        "ROC-AUC"  : roc_auc_score(y_test, y_proba),
    }
    results[name]   = metrics
    pipelines[name] = pipe

    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['No Churn','Churn'])}")

# ─────────────────────────────────────────────────────────
# 7. PICK BEST MODEL
# ─────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["ROC-AUC"])
best_pipe  = pipelines[best_name]
print(f"\n★  Best Model (by ROC-AUC): {best_name} — {results[best_name]['ROC-AUC']:.4f}")

# Save best model
model_path = os.path.join(MODEL_DIR, "best_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best_pipe, f)
print(f"   Saved → {model_path}")

# Also save feature column lists (needed by Streamlit app)
meta = {"num_features": num_features, "cat_features": cat_features,
        "best_model_name": best_name}
with open(os.path.join(MODEL_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

# ─────────────────────────────────────────────────────────
# 8. PLOT A — Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(results).T

fig, ax = plt.subplots(figsize=(12, 6))
x     = np.arange(len(metrics_df))
width = 0.15
colors = ["#3498DB","#2ECC71","#E74C3C","#F39C12","#9B59B6"]
metric_names = metrics_df.columns.tolist()

for i, (metric, color) in enumerate(zip(metric_names, colors)):
    bars = ax.bar(x + i * width, metrics_df[metric], width,
                  label=metric, color=color, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=7.5, rotation=90)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(metrics_df.index, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — All Metrics", fontsize=14, fontweight="bold")
ax.legend(loc="upper right")
save(fig, "08_model_comparison.png")

# ─────────────────────────────────────────────────────────
# 9. PLOT B — ROC Curves (all models)
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
line_colors = ["#3498DB","#E74C3C","#2ECC71"]

for (name, pipe), color in zip(pipelines.items(), line_colors):
    y_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
save(fig, "09_roc_curves.png")

# ─────────────────────────────────────────────────────────
# 10. PLOT C — Confusion Matrix (best model)
# ─────────────────────────────────────────────────────────
y_pred_best = best_pipe.predict(X_test)
cm          = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn","Churn"],
            yticklabels=["No Churn","Churn"],
            linewidths=1, ax=ax)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual",    fontsize=12)
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
save(fig, "10_confusion_matrix.png")

# ─────────────────────────────────────────────────────────
# 11. PLOT D — Feature Importances (Random Forest / GB)
# ─────────────────────────────────────────────────────────
for name in ["Gradient Boosting", "Random Forest"]:
    if name in pipelines:
        pipe  = pipelines[name]
        clf   = pipe.named_steps["clf"]
        ohe   = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
        cat_names = ohe.get_feature_names_out(cat_features).tolist()
        all_names = num_features + cat_names
        importances = clf.feature_importances_
        top_n = 20
        idx = np.argsort(importances)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors_bar = ["#E74C3C" if i < 5 else "#3498DB" for i in range(top_n)]
        ax.barh([all_names[i] for i in idx][::-1],
                importances[idx][::-1],
                color=colors_bar[::-1], edgecolor="white")
        ax.set_title(f"Top {top_n} Feature Importances — {name}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance")
        save(fig, f"11_feature_importance_{name.replace(' ','_').lower()}.png")
        break

# ─────────────────────────────────────────────────────────
# 12. CROSS-VALIDATION (best model)
# ─────────────────────────────────────────────────────────
print(f"\n── 5-Fold Cross-Validation ({best_name}) ──")
cv_scores = cross_val_score(best_pipe, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"  ROC-AUC per fold: {cv_scores.round(4)}")
print(f"  Mean: {cv_scores.mean():.4f}  |  Std: {cv_scores.std():.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, 6), cv_scores, color="#3498DB", edgecolor="white", width=0.5)
ax.axhline(cv_scores.mean(), color="#E74C3C", linestyle="--", lw=2,
           label=f"Mean = {cv_scores.mean():.3f}")
ax.set_xticks(range(1, 6))
ax.set_xticklabels([f"Fold {i}" for i in range(1, 6)])
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("ROC-AUC")
ax.set_title(f"5-Fold CV ROC-AUC — {best_name}", fontsize=13, fontweight="bold")
ax.legend()
save(fig, "12_cross_validation.png")

print("\n" + "=" * 55)
print("  ML PIPELINE COMPLETE")
print("=" * 55)
print(f"  Best model : {best_name}")
print(f"  ROC-AUC    : {results[best_name]['ROC-AUC']:.4f}")
print(f"  F1 Score   : {results[best_name]['F1']:.4f}")
print(f"  Saved to   : {MODEL_DIR}/best_model.pkl")
print("\n  ➜  Run: streamlit run churn_app.py")
