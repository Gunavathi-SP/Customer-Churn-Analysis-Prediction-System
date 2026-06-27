"""
=============================================================
  Project 2 — Customer Churn Prediction
  Script 3: Streamlit Web App
=============================================================
Run:
    streamlit run churn_app.py

Prerequisites:
    pip install streamlit plotly pandas scikit-learn xgboost
    Run churn_ml.py first to generate models/best_model.pkl
=============================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (roc_curve, roc_auc_score,
                             confusion_matrix, classification_report)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 4px solid #E74C3C;
        margin-bottom: 1rem;
    }
    .churn-yes {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
    }
    .churn-no {
        background: linear-gradient(135deg, #55efc4, #00b894);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔄 Customer Churn Prediction</h1>
    <p style="font-size:1.1rem; opacity:0.85;">
        ML-powered early warning system · EDA · Model Evaluation · Live Prediction
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
COLORS = {"Yes": "#E74C3C", "No": "#2ECC71"}

@st.cache_data
def clean_data(df_raw):
    df = df_raw.copy()
    df["Gender"]   = df["Gender"].str.strip().replace({"M": "Male",  "F": "Female"})
    df["IsActive"] = df["IsActive"].str.strip().replace({"Y": "Yes", "N": "No"})
    df["Country"]  = df["Country"].str.strip().replace({"IND": "India", "IN": "India"})
    df["DiscountUsed"] = pd.to_numeric(df["DiscountUsed"], errors="coerce")
    cat_cols = ["Gender", "ProductCategory", "PaymentMethod",
                "IsActive", "Browser", "Device", "Country"]
    for c in cat_cols:
        if c in df.columns:
            df[c].fillna(df[c].mode()[0], inplace=True)
    for c in ["Returns", "ReviewScore", "DiscountUsed"]:
        if c in df.columns:
            df[c].fillna(df[c].median(), inplace=True)
    return df

@st.cache_resource
def load_model():
    path = os.path.join("models", "best_model.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_meta():
    path = os.path.join("models", "meta.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=80)
    st.markdown("## 📂 Upload Dataset")
    uploaded = st.file_uploader("Upload churn_prediction.csv", type=["csv"])

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "This app performs end-to-end customer churn analysis:\n\n"
        "• **EDA** — Data exploration & visualizations\n"
        "• **Model** — ML performance metrics\n"
        "• **Predict** — Single customer risk score"
    )
    st.markdown("---")
    meta = load_meta()
    if meta:
        st.success(f"✅ Model loaded: **{meta.get('best_model_name','—')}**")

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
if uploaded:
    df_raw = pd.read_csv(uploaded)
else:
    default = "churn_prediction.csv"
    if os.path.exists(default):
        df_raw = pd.read_csv(default)
        st.info("📊 Using default `churn_prediction.csv`. Upload your own via the sidebar.")
    else:
        st.warning("⬆️ Please upload `churn_prediction.csv` using the sidebar.")
        st.stop()

df = clean_data(df_raw)

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA & Insights",
    "🤖 Model Performance",
    "🎯 Predict Single Customer",
    "📋 Dataset Explorer",
])

# ═══════════════════════════════════════════════════════
# TAB 1 — EDA
# ═══════════════════════════════════════════════════════
with tab1:
    # KPI Row
    total   = len(df)
    churned = (df["Churn"] == "Yes").sum()
    retained= total - churned
    churn_r = churned / total * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers",  f"{total:,}")
    k2.metric("Churned",          f"{churned:,}",   f"{churn_r:.1f}%")
    k3.metric("Retained",         f"{retained:,}",  f"{100-churn_r:.1f}%")
    k4.metric("Avg Session Time", f"{df['SessionTime'].mean():.0f} min")

    st.markdown('<div class="section-header">Churn Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        counts = df["Churn"].value_counts().reset_index()
        counts.columns = ["Churn", "Count"]
        fig = px.pie(counts, names="Churn", values="Count",
                     color="Churn", color_discrete_map=COLORS,
                     title="Churn Share", hole=0.4)
        fig.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(counts, x="Churn", y="Count",
                     color="Churn", color_discrete_map=COLORS,
                     title="Churn Count", text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Numeric distributions
    st.markdown('<div class="section-header">Numeric Feature Distributions by Churn</div>',
                unsafe_allow_html=True)
    num_feat = st.selectbox("Select feature", ["Age","Income","SpendingScore",
                                                "PurchaseAmount","SessionTime",
                                                "ReviewScore","Returns","DiscountUsed"])
    fig = px.histogram(df, x=num_feat, color="Churn",
                       color_discrete_map=COLORS, barmode="overlay",
                       opacity=0.7, marginal="box",
                       title=f"{num_feat} Distribution by Churn")
    st.plotly_chart(fig, use_container_width=True)

    # Categorical churn rates
    st.markdown('<div class="section-header">Churn Rate by Category</div>',
                unsafe_allow_html=True)
    cat_sel = st.selectbox("Select category", ["Gender","ProductCategory",
                                                "PaymentMethod","IsActive",
                                                "Device","Browser"])
    rate_df = (
        df.groupby(cat_sel)["Churn"]
          .apply(lambda x: (x == "Yes").mean() * 100)
          .reset_index()
    )
    rate_df.columns = [cat_sel, "ChurnRate"]
    rate_df = rate_df.sort_values("ChurnRate", ascending=False)
    fig = px.bar(rate_df, x=cat_sel, y="ChurnRate",
                 color="ChurnRate", color_continuous_scale="RdYlGn_r",
                 title=f"Churn Rate (%) by {cat_sel}",
                 text=rate_df["ChurnRate"].round(1).astype(str) + "%")
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-header">Correlation Heatmap</div>',
                unsafe_allow_html=True)
    df_enc = df.copy()
    df_enc["Churn_bin"] = (df_enc["Churn"] == "Yes").astype(int)
    corr_cols = ["Age","Income","SpendingScore","PurchaseAmount",
                 "Returns","DiscountUsed","ReviewScore","SessionTime","Churn_bin"]
    corr = df_enc[corr_cols].corr().round(3)
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                    color_continuous_scale="RdYlGn",
                    title="Correlation Matrix (with Churn)")
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════
with tab2:
    model = load_model()

    if model is None:
        st.warning("⚠️ No trained model found. Please run `churn_ml.py` first.")
    else:
        st.markdown('<div class="section-header">Test Set Performance</div>',
                    unsafe_allow_html=True)

        num_features = ["Age","Income","SpendingScore","PurchaseAmount",
                        "Returns","DiscountUsed","ReviewScore","SessionTime"]
        cat_features = ["Gender","ProductCategory","PaymentMethod",
                        "IsActive","Browser","Device","Country"]

        df2 = df.copy()
        df2["Churn_bin"] = (df2["Churn"] == "Yes").astype(int)
        feat_cols = num_features + cat_features
        available = [c for c in feat_cols if c in df2.columns]
        X = df2[available]
        y = df2["Churn_bin"]

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, roc_curve, confusion_matrix)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Accuracy",  f"{acc:.3f}")
        m2.metric("Precision", f"{pre:.3f}")
        m3.metric("Recall",    f"{rec:.3f}")
        m4.metric("F1 Score",  f"{f1:.3f}")
        m5.metric("ROC-AUC",   f"{auc:.3f}")

        col1, col2 = st.columns(2)

        # ROC Curve
        with col1:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"Model (AUC={auc:.3f})",
                                     line=dict(color="#3498DB", width=3)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     name="Baseline",
                                     line=dict(color="gray", dash="dash")))
            fig.update_layout(title="ROC Curve",
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)

        # Confusion Matrix
        with col2:
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True,
                            x=["Predicted: No","Predicted: Yes"],
                            y=["Actual: No",   "Actual: Yes"],
                            color_continuous_scale="Blues",
                            title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)

        # Prediction distribution
        st.markdown('<div class="section-header">Churn Probability Distribution</div>',
                    unsafe_allow_html=True)
        prob_df = pd.DataFrame({"Probability": y_proba, "Actual": y_test.values})
        prob_df["Actual_label"] = prob_df["Actual"].map({0:"No Churn",1:"Churn"})
        fig = px.histogram(prob_df, x="Probability", color="Actual_label",
                           color_discrete_map={"No Churn":"#2ECC71","Churn":"#E74C3C"},
                           barmode="overlay", opacity=0.7,
                           title="Predicted Churn Probability by True Label",
                           nbins=40)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 3 — SINGLE CUSTOMER PREDICTION
# ═══════════════════════════════════════════════════════
with tab3:
    model = load_model()
    if model is None:
        st.warning("⚠️ Run `churn_ml.py` first to enable predictions.")
    else:
        st.markdown("### Enter Customer Details")
        st.markdown("Fill in the customer's profile below and click **Predict**.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            age    = st.slider("Age", 18, 70, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            income = st.number_input("Annual Income (₹)", 10000, 200000, 50000, step=1000)
            country= st.selectbox("Country", ["India"])

        with col2:
            st.markdown("**Purchase Behaviour**")
            spending_score = st.slider("Spending Score (1-100)", 1, 100, 50)
            purchase_amt   = st.number_input("Purchase Amount (₹)", 100, 50000, 5000, step=100)
            category       = st.selectbox("Product Category", ["Electronics","Clothing","Beauty","Home"])
            payment        = st.selectbox("Payment Method", ["UPI","Card","Cash","NetBanking"])
            discount       = st.slider("Discount Used (%)", 0, 100, 20)
            returns        = st.slider("Number of Returns", 0.0, 10.0, 1.0, step=0.5)

        with col3:
            st.markdown("**Engagement**")
            is_active    = st.selectbox("Is Active?", ["Yes", "No"])
            review_score = st.slider("Review Score", 1.0, 5.0, 3.0, step=0.1)
            session_time = st.slider("Session Time (min)", 1, 300, 120)
            browser      = st.selectbox("Browser", ["Chrome","Firefox","Edge"])
            device       = st.selectbox("Device", ["Mobile","Desktop","Tablet"])

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary")

        if predict_btn:
            input_df = pd.DataFrame([{
                "Age": age, "Gender": gender, "Income": income,
                "SpendingScore": spending_score, "PurchaseAmount": purchase_amt,
                "ProductCategory": category, "PaymentMethod": payment,
                "IsActive": is_active, "Returns": returns,
                "DiscountUsed": discount, "ReviewScore": review_score,
                "Browser": browser, "Device": device, "SessionTime": session_time,
                "Country": country,
            }])

            prob_churn = model.predict_proba(input_df)[0][1]
            pred_label = "Yes" if prob_churn >= 0.5 else "No"

            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if pred_label == "Yes":
                    st.markdown(f"""
                    <div class="churn-yes">
                        ⚠️ HIGH CHURN RISK<br>
                        <span style="font-size:2rem">{prob_churn*100:.1f}%</span><br>
                        <span style="font-size:0.9rem">probability of churning</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="churn-no">
                        ✅ LOW CHURN RISK<br>
                        <span style="font-size:2rem">{prob_churn*100:.1f}%</span><br>
                        <span style="font-size:0.9rem">probability of churning</span>
                    </div>
                    """, unsafe_allow_html=True)

            with res_col2:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob_churn * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Churn Probability (%)"},
                    delta={"reference": 50},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#E74C3C" if pred_label=="Yes" else "#2ECC71"},
                        "steps": [
                            {"range": [0,  40], "color": "#d5f5e3"},
                            {"range": [40, 60], "color": "#fef9e7"},
                            {"range": [60, 100],"color": "#fadbd8"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                ))
                fig.update_layout(height=280, margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            # Recommendations
            st.markdown("#### 💡 Retention Recommendations")
            tips = []
            if prob_churn >= 0.5:
                if is_active == "No":
                    tips.append("📧 Send a re-engagement email with a personalised offer.")
                if discount < 20:
                    tips.append("🎁 Offer a loyalty discount or coupon (current usage is low).")
                if review_score < 3:
                    tips.append("📞 Reach out to address negative experience (low review score).")
                if returns > 3:
                    tips.append("🔄 Investigate high return rate — possible product quality issue.")
                if session_time < 60:
                    tips.append("📱 Push personalised product recommendations to increase engagement.")
                if not tips:
                    tips.append("🤝 Assign a dedicated account manager for proactive retention.")
            else:
                tips.append("✅ Customer is healthy. Continue regular engagement.")
                tips.append("🌟 Consider enrolling in loyalty rewards programme.")
            for tip in tips:
                st.markdown(f"- {tip}")

# ═══════════════════════════════════════════════════════
# TAB 4 — DATASET EXPLORER
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows",    f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())

    st.markdown("**Filter by Churn**")
    churn_filter = st.radio("Show:", ["All","Churned","Retained"], horizontal=True)
    if churn_filter == "Churned":
        view = df[df["Churn"] == "Yes"]
    elif churn_filter == "Retained":
        view = df[df["Churn"] == "No"]
    else:
        view = df

    st.dataframe(view.head(200), use_container_width=True)

    st.markdown("**Descriptive Statistics**")
    st.dataframe(df.describe(include="all").round(2), use_container_width=True)
