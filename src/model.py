"""
Stage 3: Merchant Churn Modeling
1. Cox Proportional Hazards — time-to-churn survival analysis
2. XGBoost Churn Classifier — predict at-risk merchants
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, classification_report,
    roc_curve, confusion_matrix
)
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# ── 1. LOAD & PREPARE DATA ────────────────────────────────────────────────────

print("Loading merchant features...")
df = pd.read_csv("data/processed/merchant_features.csv")

# Fill nulls in review columns with median
df["avg_review_score"]       = df["avg_review_score"].fillna(df["avg_review_score"].median())
df["total_reviews"]          = df["total_reviews"].fillna(0)
df["avg_delivery_delay_days"]= df["avg_delivery_delay_days"].fillna(0)

# Encode top_category as integer for XGBoost
le = LabelEncoder()
df["category_encoded"] = le.fit_transform(df["top_category"].astype(str))

print(f"  Merchants: {len(df):,} | Churned: {df['churned'].sum():,} ({df['churned'].mean()*100:.1f}%)")

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# ── 2. KAPLAN-MEIER SURVIVAL CURVE ───────────────────────────────────────────
# Overall survival curve — how long do merchants stay active?

print("\nFitting Kaplan-Meier survival curve...")

kmf = KaplanMeierFitter()
kmf.fit(
    durations  = df["tenure_days"],
    event_observed = df["churned"],
    label      = "All Merchants"
)

# KM by review score segment (high vs low)
df["high_review"] = (df["avg_review_score"] >= df["avg_review_score"].median()).astype(int)

kmf_high = KaplanMeierFitter()
kmf_low  = KaplanMeierFitter()

kmf_high.fit(
    df.loc[df["high_review"]==1, "tenure_days"],
    df.loc[df["high_review"]==1, "churned"],
    label="High Review Score"
)
kmf_low.fit(
    df.loc[df["high_review"]==0, "tenure_days"],
    df.loc[df["high_review"]==0, "churned"],
    label="Low Review Score"
)

# Save KM data for dashboard
km_overall = kmf.survival_function_.reset_index()
km_overall.columns = ["timeline", "survival_prob"]

km_high = kmf_high.survival_function_.reset_index()
km_high.columns = ["timeline", "survival_prob"]
km_high["segment"] = "High Review Score"

km_low = kmf_low.survival_function_.reset_index()
km_low.columns = ["timeline", "survival_prob"]
km_low["segment"] = "Low Review Score"

km_segments = pd.concat([km_high, km_low])
km_overall.to_csv("data/processed/km_overall.csv", index=False)
km_segments.to_csv("data/processed/km_segments.csv", index=False)

print("  Kaplan-Meier curves saved.")

# ── 3. COX PROPORTIONAL HAZARDS MODEL ────────────────────────────────────────

print("\nFitting Cox Proportional Hazards model...")

cox_features = [
    "total_orders",
    "avg_order_value",
    "avg_review_score",
    "avg_delivery_delay_days",
    "unique_products",
    "total_reviews",
    "tenure_days",
    "churned"
]

cox_df = df[cox_features].copy()

# Clip extreme outliers for Cox stability
for col in ["total_orders", "avg_order_value", "unique_products", "total_reviews"]:
    upper = cox_df[col].quantile(0.99)
    cox_df[col] = cox_df[col].clip(upper=upper)

cph = CoxPHFitter(penalizer=0.1)
cph.fit(
    cox_df,
    duration_col   = "tenure_days",
    event_col      = "churned"
)

print("\n── Cox PH Summary ───────────────────────────────────────")
cph.print_summary(columns=["coef", "exp(coef)", "p"])

c_index = concordance_index(df["tenure_days"], -cph.predict_partial_hazard(cox_df), df["churned"])
print(f"\n  C-index (concordance): {c_index:.3f}")
print("  (>0.7 is good; >0.8 is strong)")

# Save hazard ratios for dashboard
hr_df = cph.summary[["coef", "exp(coef)", "p"]].reset_index()
hr_df.columns = ["feature", "coef", "hazard_ratio", "p_value"]
hr_df["significant"] = hr_df["p_value"] < 0.05
hr_df = hr_df.sort_values("hazard_ratio", ascending=False)
hr_df.to_csv("data/processed/hazard_ratios.csv", index=False)
print("\n  Hazard ratios saved.")

# ── 4. XGBOOST CHURN CLASSIFIER ───────────────────────────────────────────────

print("\nTraining XGBoost churn classifier...")

FEATURES = [
    "total_orders",
    "total_revenue",
    "avg_order_value",
    "unique_products",
    "days_active",
    "avg_review_score",
    "total_reviews",
    "avg_delivery_delay_days",
    "category_encoded"
]

X = df[FEATURES]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = (y == 0).sum() / (y == 1).sum()

xgb = XGBClassifier(
    n_estimators       = 300,
    max_depth          = 4,
    learning_rate      = 0.05,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    scale_pos_weight   = scale_pos_weight,
    use_label_encoder  = False,
    eval_metric        = "logloss",
    random_state       = 42
)

xgb.fit(
    X_train, y_train,
    eval_set           = [(X_test, y_test)],
    verbose            = False
)

# Evaluate
y_pred_proba = xgb.predict_proba(X_test)[:, 1]
y_pred       = (y_pred_proba >= 0.5).astype(int)
auc          = roc_auc_score(y_test, y_pred_proba)

print(f"\n── XGBoost Results ──────────────────────────────────────")
print(f"  ROC-AUC:  {auc:.3f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Active','Churned'])}")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb, X, y, cv=cv, scoring="roc_auc")
print(f"  5-Fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── 5. FEATURE IMPORTANCE ────────────────────────────────────────────────────

importance_df = pd.DataFrame({
    "feature"   : FEATURES,
    "importance": xgb.feature_importances_
}).sort_values("importance", ascending=False)

# Map category_encoded back to readable name
importance_df["feature"] = importance_df["feature"].replace(
    {"category_encoded": "top_category"}
)

importance_df.to_csv("data/processed/feature_importance.csv", index=False)

print(f"\n── Top Churn Drivers ────────────────────────────────────")
for _, row in importance_df.iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"  {row['feature']:<30} {bar} {row['importance']:.3f}")

# ── 6. SCORE ALL MERCHANTS ───────────────────────────────────────────────────

df["churn_probability"] = xgb.predict_proba(df[FEATURES])[:, 1]
df["risk_tier"] = pd.cut(
    df["churn_probability"],
    bins   = [0, 0.33, 0.66, 1.0],
    labels = ["Low Risk", "Medium Risk", "High Risk"]
)

# Top 3 risk factors per merchant from feature importance
top_features = importance_df["feature"].tolist()[:3]
df["top_risk_factors"] = ", ".join(top_features)

# Save scored merchants
scored = df[[
    "seller_id", "seller_state", "total_orders", "total_revenue",
    "avg_review_score", "avg_delivery_delay_days", "top_category",
    "tenure_days", "churned", "churn_probability", "risk_tier", "top_risk_factors"
]].copy()

scored.to_csv("data/processed/scored_merchants.csv", index=False)

print(f"\n── Risk Tier Distribution ───────────────────────────────")
print(scored["risk_tier"].value_counts().to_string())

# ── 7. ROC CURVE DATA ────────────────────────────────────────────────────────

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
roc_df.to_csv("data/processed/roc_curve.csv", index=False)

# ── 8. SAVE MODEL ────────────────────────────────────────────────────────────

with open("outputs/models/xgb_churn.pkl", "wb") as f:
    pickle.dump(xgb, f)

with open("outputs/models/cox_model.pkl", "wb") as f:
    pickle.dump(cph, f)

with open("outputs/models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print(f"\n  Models saved to outputs/models/")
print(f"\nStage 3 complete. Ready for Stage 4 — Streamlit dashboard.")
