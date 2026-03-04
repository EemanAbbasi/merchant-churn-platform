"""
Stage 4: Merchant Churn Intelligence Platform
Streamlit Dashboard — 4 pages:
1. Overview
2. Survival Analysis
3. Churn Drivers
4. At-Risk Merchants
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "Merchant Churn Intelligence",
    page_icon   = "🛒",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── STYLES ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        text-align: center;
        overflow: hidden;
    }
    .metric-value { font-size: 1.4rem; font-weight: 700; color: #1a5276; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .metric-label { font-size: 0.78rem; color: #666; margin-top: 4px; }
    .risk-high   { color: #e74c3c; font-weight: 600; }
    .risk-medium { color: #f39c12; font-weight: 600; }
    .risk-low    { color: #27ae60; font-weight: 600; }
    h1 { color: #1a5276; }
    h2 { color: #1a5276; }
    .stSelectbox label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ──────────────────────────────────────────────────────────────

import os

# Base path — works locally and on Streamlit Cloud
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_data
def load_data():
    scored       = pd.read_csv(BASE_DIR / "data/processed/scored_merchants.csv")
    km_overall   = pd.read_csv(BASE_DIR / "data/processed/km_overall.csv")
    km_segments  = pd.read_csv(BASE_DIR / "data/processed/km_segments.csv")
    hazard_ratios= pd.read_csv(BASE_DIR / "data/processed/hazard_ratios.csv")
    feat_imp     = pd.read_csv(BASE_DIR / "data/processed/feature_importance.csv")
    roc_curve    = pd.read_csv(BASE_DIR / "data/processed/roc_curve.csv")
    return scored, km_overall, km_segments, hazard_ratios, feat_imp, roc_curve

scored, km_overall, km_segments, hazard_ratios, feat_imp, roc_curve = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
st.sidebar.title("Merchant Churn\nIntelligence Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "📈 Survival Analysis", "🔍 Churn Drivers", "⚠️ At-Risk Merchants"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Olist E-Commerce")
st.sidebar.markdown(f"**Merchants:** {len(scored):,}")
st.sidebar.markdown(f"**Model AUC:** 0.780")
st.sidebar.markdown(f"**Cox C-index:** 0.826")

# ── COLOUR PALETTE ────────────────────────────────────────────────────────────

NAVY   = "#1a5276"
BLUE   = "#2e86c1"
GREEN  = "#27ae60"
YELLOW = "#f39c12"
RED    = "#e74c3c"
GREY   = "#bdc3c7"

RISK_COLORS = {
    "Low Risk"    : GREEN,
    "Medium Risk" : YELLOW,
    "High Risk"   : RED
}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Overview":

    st.title("🛒 Merchant Churn Intelligence Platform")
    st.markdown("#### Predicting and understanding merchant churn across 2,970 sellers")
    st.markdown("---")

    # KPI row
    total      = len(scored)
    churned    = int(scored["churned"].sum())
    churn_rate = churned / total * 100
    high_risk  = int((scored["risk_tier"] == "High Risk").sum())
    avg_rev    = scored["total_revenue"].median()
    avg_review = scored["avg_review_score"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Merchants", f"{total:,}")
    with c2:
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with c3:
        st.metric("High Risk Merchants", f"{high_risk:,}")
    with c4:
        st.metric("Median Revenue", f"R${avg_rev:,.0f}")
    with c5:
        st.metric("Avg Review Score", f"{avg_review:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Risk tier donut
    with col1:
        st.subheader("Risk Tier Distribution")
        tier_counts = scored["risk_tier"].value_counts().reindex(
            ["High Risk", "Medium Risk", "Low Risk"]
        )
        fig = go.Figure(go.Pie(
            labels = tier_counts.index,
            values = tier_counts.values,
            hole   = 0.55,
            marker = dict(colors=[RED, YELLOW, GREEN]),
            textinfo = "label+percent",
            textfont_size = 13
        ))
        fig.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            paper_bgcolor="white"
        )
        fig.add_annotation(
            text=f"<b>{high_risk}</b><br>High Risk",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=RED)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Churn probability distribution
    with col2:
        st.subheader("Churn Probability Distribution")
        fig = go.Figure()
        for tier, color in RISK_COLORS.items():
            subset = scored[scored["risk_tier"] == tier]["churn_probability"]
            fig.add_trace(go.Histogram(
                x=subset, name=tier,
                marker_color=color, opacity=0.75,
                nbinsx=30
            ))
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Churn Probability",
            yaxis_title="Number of Merchants",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=40, b=40, l=40, r=20),
            height=320,
            paper_bgcolor="white",
            plot_bgcolor="#f8f9fa"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Revenue at risk
    st.subheader("Revenue at Risk by Tier")
    rev_risk = scored.groupby("risk_tier")["total_revenue"].sum().reindex(
        ["High Risk", "Medium Risk", "Low Risk"]
    ).reset_index()
    rev_risk.columns = ["risk_tier", "total_revenue"]

    fig = go.Figure(go.Bar(
        x     = rev_risk["risk_tier"],
        y     = rev_risk["total_revenue"],
        marker_color = [RED, YELLOW, GREEN],
        text  = rev_risk["total_revenue"].apply(lambda x: f"R${x:,.0f}"),
        textposition = "outside"
    ))
    fig.update_layout(
        yaxis_title="Total Revenue (R$)",
        xaxis_title="",
        margin=dict(t=60, b=40, l=60, r=20),
        height=300,
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top categories by churn rate
    st.subheader("Churn Rate by Product Category")
    cat_churn = scored.groupby("top_category").agg(
        total=("churned", "count"),
        churned=("churned", "sum")
    ).reset_index()
    cat_churn["churn_rate"] = cat_churn["churned"] / cat_churn["total"]
    cat_churn = cat_churn[cat_churn["total"] >= 20].sort_values(
        "churn_rate", ascending=True
    ).tail(15)

    fig = go.Figure(go.Bar(
        x           = cat_churn["churn_rate"] * 100,
        y           = cat_churn["top_category"],
        orientation = "h",
        marker_color= BLUE,
        text        = cat_churn["churn_rate"].apply(lambda x: f"{x*100:.1f}%"),
        textposition= "outside"
    ))
    fig.update_layout(
        xaxis_title = "Churn Rate (%)",
        margin      = dict(t=20, b=40, l=180, r=80),
        height      = 420,
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa"
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SURVIVAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Survival Analysis":

    st.title("📈 Survival Analysis")
    st.markdown("#### How long do merchants stay active — and what predicts early exit?")
    st.markdown("""
    Survival analysis models the **time until an event** (churn). 
    The survival curve shows the probability a merchant is still active at any given day.
    A steeper drop means faster churn. Segments that stay higher for longer are more retained.
    """)
    st.markdown("---")

    col1, col2 = st.columns(2)

    # Overall KM curve
    with col1:
        st.subheader("Overall Merchant Survival Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x    = km_overall["timeline"],
            y    = km_overall["survival_prob"],
            mode = "lines",
            name = "All Merchants",
            line = dict(color=NAVY, width=2.5),
            fill = "tozeroy",
            fillcolor = "rgba(26,82,118,0.08)"
        ))
        median_idx = (km_overall["survival_prob"] - 0.5).abs().idxmin()
        median_day = km_overall.loc[median_idx, "timeline"]
        fig.add_vline(
            x=median_day, line_dash="dash", line_color=RED,
            annotation_text=f"Median: {median_day:.0f} days",
            annotation_position="top right"
        )
        fig.update_layout(
            xaxis_title = "Days Since First Order",
            yaxis_title = "Survival Probability",
            yaxis       = dict(tickformat=".0%", range=[0,1]),
            margin      = dict(t=40, b=60, l=60, r=20),
            height      = 360,
            paper_bgcolor="white",
            plot_bgcolor="#f8f9fa"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"📌 50% of merchants churn within **{median_day:.0f} days** of their first order.")

    # Segmented KM curve
    with col2:
        st.subheader("Survival by Review Score Segment")
        seg_colors = {"High Review Score": GREEN, "Low Review Score": RED}
        fig = go.Figure()
        for seg in km_segments["segment"].unique():
            seg_df = km_segments[km_segments["segment"] == seg]
            fig.add_trace(go.Scatter(
                x    = seg_df["timeline"],
                y    = seg_df["survival_prob"],
                mode = "lines",
                name = seg,
                line = dict(color=seg_colors.get(seg, BLUE), width=2.5)
            ))
        fig.update_layout(
            xaxis_title = "Days Since First Order",
            yaxis_title = "Survival Probability",
            yaxis       = dict(tickformat=".0%", range=[0,1]),
            legend      = dict(orientation="h", yanchor="bottom", y=1.02),
            margin      = dict(t=40, b=60, l=60, r=20),
            height      = 360,
            paper_bgcolor="white",
            plot_bgcolor="#f8f9fa"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 Merchants with higher review scores retain significantly longer.")

    st.markdown("---")

    # Cox PH results
    st.subheader("Cox Proportional Hazards — What Accelerates Churn?")
    st.markdown("""
    The Cox model estimates each feature's effect on churn **hazard** (risk over time).
    - **Hazard Ratio > 1** → increases churn risk
    - **Hazard Ratio < 1** → decreases churn risk (protective)
    - Only statistically significant features (p < 0.05) are shown filled.
    """)

    hr_plot = hazard_ratios.copy()
    hr_plot["color"] = hr_plot.apply(
        lambda r: RED if r["hazard_ratio"] > 1 else GREEN, axis=1
    )
    hr_plot["opacity"] = hr_plot["significant"].apply(lambda x: 1.0 if x else 0.35)

    fig = go.Figure()
    for _, row in hr_plot.iterrows():
        fig.add_trace(go.Bar(
            x           = [row["hazard_ratio"]],
            y           = [row["feature"]],
            orientation = "h",
            marker_color= row["color"],
            opacity     = row["opacity"],
            name        = row["feature"],
            showlegend  = False,
            hovertemplate = (
                f"<b>{row['feature']}</b><br>"
                f"Hazard Ratio: {row['hazard_ratio']:.3f}<br>"
                f"p-value: {row['p_value']:.4f}<extra></extra>"
            )
        ))

    fig.add_vline(x=1.0, line_dash="dash", line_color=GREY, line_width=1.5,
                  annotation_text="Baseline (HR=1)", annotation_position="top")
    fig.update_layout(
        xaxis_title = "Hazard Ratio",
        margin      = dict(t=40, b=60, l=200, r=60),
        height      = 340,
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa"
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cox C-index", "0.826", help="Model discriminates churn risk correctly 82.6% of the time")
    with col2:
        st.metric("Events Observed", "812", help="Merchants confirmed churned in the dataset")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CHURN DRIVERS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Churn Drivers":

    st.title("🔍 Churn Drivers")
    st.markdown("#### What features most strongly predict merchant churn?")
    st.markdown("---")

    col1, col2 = st.columns(2)

    # Feature importance
    with col1:
        st.subheader("XGBoost Feature Importance")
        st.markdown("Relative contribution of each feature to churn prediction.")
        fig = go.Figure(go.Bar(
            x           = feat_imp["importance"],
            y           = feat_imp["feature"],
            orientation = "h",
            marker_color= [
                RED if i == 0 else BLUE if i <= 2 else GREY
                for i in range(len(feat_imp))
            ],
            text        = feat_imp["importance"].apply(lambda x: f"{x:.3f}"),
            textposition= "outside"
        ))
        fig.update_layout(
            xaxis_title = "Feature Importance Score",
            margin      = dict(t=20, b=40, l=180, r=60),
            height      = 380,
            paper_bgcolor="white",
            plot_bgcolor="#f8f9fa",
            yaxis       = dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)

    # ROC curve
    with col2:
        st.subheader("XGBoost ROC Curve")
        st.markdown("Model's ability to distinguish churned vs active merchants.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x    = roc_curve["fpr"],
            y    = roc_curve["tpr"],
            mode = "lines",
            name = "XGBoost (AUC = 0.780)",
            line = dict(color=NAVY, width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1],
            mode="lines",
            name="Random Baseline",
            line=dict(color=GREY, dash="dash", width=1.5)
        ))
        fig.update_layout(
            xaxis_title = "False Positive Rate",
            yaxis_title = "True Positive Rate",
            legend      = dict(orientation="h", yanchor="bottom", y=1.02),
            margin      = dict(t=40, b=60, l=60, r=20),
            height      = 380,
            paper_bgcolor="white",
            plot_bgcolor="#f8f9fa"
        )
        fig.add_annotation(
            x=0.65, y=0.25,
            text="<b>AUC = 0.780</b>",
            showarrow=False,
            font=dict(size=14, color=NAVY),
            bgcolor="white",
            bordercolor=NAVY,
            borderwidth=1
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Scatter: review score vs churn probability
    st.subheader("Review Score vs Churn Probability")
    fig = px.scatter(
        scored,
        x           = "avg_review_score",
        y           = "churn_probability",
        color       = "risk_tier",
        color_discrete_map = RISK_COLORS,
        size        = "total_orders",
        size_max    = 18,
        opacity     = 0.65,
        hover_data  = ["seller_state", "top_category", "total_revenue"],
        labels      = {
            "avg_review_score"  : "Average Review Score",
            "churn_probability" : "Churn Probability",
            "risk_tier"         : "Risk Tier"
        }
    )
    fig.update_layout(
        margin      = dict(t=20, b=60, l=60, r=20),
        height      = 400,
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Delivery delay vs churn probability
    st.subheader("Delivery Delay vs Churn Probability")
    st.markdown("Positive delay = delivered late. Negative = delivered early.")
    fig = px.scatter(
        scored[scored["avg_delivery_delay_days"].between(-30, 30)],
        x           = "avg_delivery_delay_days",
        y           = "churn_probability",
        color       = "risk_tier",
        color_discrete_map = RISK_COLORS,
        opacity     = 0.6,
        labels      = {
            "avg_delivery_delay_days": "Avg Delivery Delay (days)",
            "churn_probability"      : "Churn Probability",
            "risk_tier"              : "Risk Tier"
        }
    )
    fig.add_vline(x=0, line_dash="dash", line_color=GREY,
                  annotation_text="On-time", annotation_position="top")
    fig.update_layout(
        margin      = dict(t=20, b=60, l=60, r=20),
        height      = 380,
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa"
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — AT-RISK MERCHANTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚠️ At-Risk Merchants":

    st.title("⚠️ At-Risk Merchants")
    st.markdown("#### Identify and prioritize merchants most likely to churn")
    st.markdown("---")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect(
            "Risk Tier",
            options = ["High Risk", "Medium Risk", "Low Risk"],
            default = ["High Risk"]
        )
    with col2:
        state_options = ["All"] + sorted(scored["seller_state"].unique().tolist())
        state_filter  = st.selectbox("State", state_options)
    with col3:
        cat_options = ["All"] + sorted(scored["top_category"].dropna().unique().tolist())
        cat_filter  = st.selectbox("Category", cat_options)

    # Apply filters
    filtered = scored[scored["risk_tier"].isin(risk_filter)]
    if state_filter != "All":
        filtered = filtered[filtered["seller_state"] == state_filter]
    if cat_filter != "All":
        filtered = filtered[filtered["top_category"] == cat_filter]

    filtered = filtered.sort_values("churn_probability", ascending=False)

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Merchants Shown", f"{len(filtered):,}")
    with c2:
        st.metric("Revenue at Risk", f"R${filtered['total_revenue'].sum():,.0f}")
    with c3:
        st.metric("Avg Churn Probability", f"{filtered['churn_probability'].mean()*100:.1f}%")
    with c4:
        st.metric("Avg Review Score", f"{filtered['avg_review_score'].mean():.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Churn probability bar chart for top merchants
    st.subheader(f"Top 20 Highest-Risk Merchants")
    top20 = filtered.head(20).copy()
    top20["seller_short"] = top20["seller_id"].str[:8] + "..."

    fig = go.Figure(go.Bar(
        x           = top20["churn_probability"],
        y           = top20["seller_short"],
        orientation = "h",
        marker_color= top20["risk_tier"].map(RISK_COLORS),
        text        = top20["churn_probability"].apply(lambda x: f"{x*100:.1f}%"),
        textposition= "outside",
        hovertemplate=(
            "<b>Seller:</b> %{customdata[0]}<br>"
            "<b>State:</b> %{customdata[1]}<br>"
            "<b>Category:</b> %{customdata[2]}<br>"
            "<b>Revenue:</b> R$%{customdata[3]:,.0f}<br>"
            "<b>Churn Prob:</b> %{x:.1%}<extra></extra>"
        ),
        customdata=top20[["seller_id","seller_state","top_category","total_revenue"]].values
    ))
    fig.update_layout(
        xaxis_title = "Churn Probability",
        xaxis       = dict(tickformat=".0%", range=[0, 1.1]),
        margin      = dict(t=20, b=40, l=120, r=80),
        height      = 480,
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa",
        yaxis       = dict(autorange="reversed")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Full table
    st.subheader("Full Merchant Table")
    display_cols = {
        "seller_id"              : "Seller ID",
        "seller_state"           : "State",
        "top_category"           : "Category",
        "total_orders"           : "Orders",
        "total_revenue"          : "Revenue (R$)",
        "avg_review_score"       : "Review Score",
        "avg_delivery_delay_days": "Delivery Delay (d)",
        "tenure_days"            : "Tenure (days)",
        "churn_probability"      : "Churn Prob",
        "risk_tier"              : "Risk Tier",
        "top_risk_factors"       : "Top Risk Factors"
    }

    display_df = filtered[list(display_cols.keys())].rename(columns=display_cols).copy()
    display_df["Churn Prob"]     = display_df["Churn Prob"].apply(lambda x: f"{x*100:.1f}%")
    display_df["Revenue (R$)"]   = display_df["Revenue (R$)"].apply(lambda x: f"{x:,.0f}")
    display_df["Review Score"]   = display_df["Review Score"].apply(lambda x: f"{x:.2f}")

    st.dataframe(display_df, use_container_width=True, height=420)

    # Download button
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "⬇️ Download At-Risk Merchant List",
        data      = csv,
        file_name = "at_risk_merchants.csv",
        mime      = "text/csv"
    )

    st.markdown("---")
    st.markdown("""
    **How to use this table:**
    - **High Risk** merchants warrant immediate outreach — review score improvement, delivery support, or catalog expansion incentives
    - **Top Risk Factors** column shows the 3 features driving each merchant's score
    - Use the **Download** button to share with account management teams
    """)
