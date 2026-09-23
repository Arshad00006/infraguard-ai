"""
INFRAguard AI – Explainable Project Risk Intelligence
Smart India Hackathon 2026 | Team Sentinel | PS 26103
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Make sure the current folder is in the path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.data_generator import generate_projects
from utils.model_utils import (
    train_models, load_models, predict_risks,
    prepare_features, get_shap_explanation, simulate_intervention,
    FEATURE_COLS
)

# ──────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS (Pastel + Gov style)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="INFRAguard AI | Project Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Serif:wght@500;600&display=swap');

    :root {
        --primary: #1B4F72;
        --primary-light: #2980B9;
        --accent: #5DADE2;
        --pastel-blue: #D6EAF8;
        --pastel-green: #D5F5E3;
        --pastel-orange: #FDEBD0;
        --pastel-red: #FADBD8;
        --pastel-purple: #E8DAEF;
        --text-dark: #1C2833;
        --text-muted: #5D6D7E;
        --bg: #F4F7FA;
        --card: #FFFFFF;
        --border: #D5DBDB;
    }

    .stApp {
        background-color: var(--bg);
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1B4F72 0%, #2980B9 100%);
        padding: 1.2rem 2rem;
        border-radius: 0 0 12px 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(27, 79, 114, 0.25);
    }
    .main-header h1 {
        font-family: 'IBM Plex Serif', serif;
        font-size: 1.8rem;
        margin: 0;
        font-weight: 600;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* Cards */
    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        text-align: center;
        height: 100%;
    }
    .metric-card h3 {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin: 0 0 0.4rem 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .metric-card .value {
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0;
    }

    .risk-high { background: #FADBD8 !important; border-left: 4px solid #E74C3C; }
    .risk-medium { background: #FDEBD0 !important; border-left: 4px solid #E67E22; }
    .risk-low { background: #D5F5E3 !important; border-left: 4px solid #27AE60; }

    /* Section titles */
    .section-title {
        font-family: 'IBM Plex Serif', serif;
        color: var(--primary);
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1.2rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid var(--pastel-blue);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B4F72 0%, #154360 100%);
    }
    [data-testid="stSidebar"] * {
        color: #EBF5FB !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        color: #D4E6F1 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--pastel-blue);
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        color: var(--primary);
    }
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1B4F72, #2980B9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.4rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #154360, #1B4F72);
        box-shadow: 0 4px 12px rgba(27,79,114,0.3);
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: var(--text-muted);
        font-size: 0.85rem;
        margin-top: 2rem;
        border-top: 1px solid var(--border);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: var(--primary);
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA & MODEL LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_or_generate_data():
    data_path = Path("data/sample_projects.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    df = generate_projects(250)
    data_path.parent.mkdir(exist_ok=True)
    df.to_csv(data_path, index=False)
    return df


@st.cache_resource
def get_trained_models(df):
    model_dir = Path("models")
    if not (model_dir / "overall_risk_model.cbm").exists():
        with st.spinner("Training risk models (first run only)..."):
            train_models(df, str(model_dir))
    return load_models(str(model_dir))


def risk_badge(score):
    if score >= 70:
        return "🔴 High"
    elif score >= 40:
        return "🟠 Medium"
    return "🟢 Low"


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ INFRAguard AI</h1>
    <p>Explainable AI layer for Infrastructure Project Risk Intelligence &nbsp;|&nbsp; Smart India Hackathon 2026 &nbsp;|&nbsp; Team Sentinel &nbsp;|&nbsp; PS 26103</p>
</div>
""", unsafe_allow_html=True)

# Load data & models
df_raw = load_or_generate_data()
models = get_trained_models(df_raw)
df = predict_risks(df_raw, models)

# ──────────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Portfolio Filters")
    ministries = ["All"] + sorted(df["ministry"].unique().tolist())
    selected_ministry = st.selectbox("Ministry", ministries)

    sectors = ["All"] + sorted(df["sector"].unique().tolist())
    selected_sector = st.selectbox("Sector", sectors)

    status_opts = ["All"] + sorted(df["status"].unique().tolist())
    selected_status = st.selectbox("Project Status", status_opts)

    risk_filter = st.select_slider(
        "Minimum Risk Score",
        options=[0, 20, 40, 60, 80],
        value=0
    )

    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    **MONITOR → PREDICT → EXPLAIN → ACT**

    INFRAguard AI converts project monitoring data into actionable cost, schedule and implementation risk intelligence with full explainability.
    """)
    st.caption("Data is synthetic & PAIMANA-aligned for demo purposes.")

# Apply filters
filtered = df.copy()
if selected_ministry != "All":
    filtered = filtered[filtered["ministry"] == selected_ministry]
if selected_sector != "All":
    filtered = filtered[filtered["sector"] == selected_sector]
if selected_status != "All":
    filtered = filtered[filtered["status"] == selected_status]
filtered = filtered[filtered["risk_score"] >= risk_filter]

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Risk Dashboard",
    "🔍 Project Deep-Dive",
    "🧠 Explain Prediction",
    "🔄 What-If Simulator",
    "⚠️ Early Warning Queue"
])

# ══════════════════════════════════════════════
# TAB 1: RISK DASHBOARD
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Portfolio Risk Overview</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Projects</h3>
            <p class="value">{len(filtered)}</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        high = len(filtered[filtered["risk_score"] >= 70])
        st.markdown(f"""
        <div class="metric-card risk-high">
            <h3>High Risk</h3>
            <p class="value">{high}</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        med = len(filtered[(filtered["risk_score"] >= 40) & (filtered["risk_score"] < 70)])
        st.markdown(f"""
        <div class="metric-card risk-medium">
            <h3>Medium Risk</h3>
            <p class="value">{med}</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        low = len(filtered[filtered["risk_score"] < 40])
        st.markdown(f"""
        <div class="metric-card risk-low">
            <h3>Low Risk</h3>
            <p class="value">{low}</p>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        avg = filtered["risk_score"].mean() if len(filtered) else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Risk Score</h3>
            <p class="value">{avg:.0f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown("##### Risk Distribution by Sector")
        sector_risk = filtered.groupby("sector")["risk_score"].mean().sort_values(ascending=False).reset_index()
        fig1 = px.bar(
            sector_risk.head(12), x="risk_score", y="sector", orientation="h",
            color="risk_score", color_continuous_scale=["#D5F5E3", "#FDEBD0", "#FADBD8"],
            labels={"risk_score": "Avg Risk Score", "sector": ""}
        )
        fig1.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, font=dict(family="IBM Plex Sans")
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        st.markdown("##### Risk Score Distribution")
        fig2 = px.histogram(
            filtered, x="risk_score", nbins=20,
            color_discrete_sequence=["#5DADE2"]
        )
        fig2.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Risk Score", yaxis_title="Count",
            font=dict(family="IBM Plex Sans")
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-title">High-Risk Projects (Priority Triage)</p>', unsafe_allow_html=True)
    high_risk_df = filtered[filtered["risk_score"] >= 60].sort_values("risk_score", ascending=False)
    display_cols = [
        "project_id", "project_name", "ministry", "sector", "status",
        "physical_progress_pct", "cost_overrun_pct", "milestone_slip_months",
        "risk_score", "cost_risk_prob", "delay_risk_prob"
    ]
    st.dataframe(
        high_risk_df[display_cols].head(25).style.background_gradient(
            subset=["risk_score"], cmap="RdYlGn_r"
        ),
        use_container_width=True, height=360
    )

# ══════════════════════════════════════════════
# TAB 2: PROJECT DEEP-DIVE
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Project Deep-Dive</p>', unsafe_allow_html=True)

    project_ids = filtered["project_id"].tolist()
    if not project_ids:
        st.warning("No projects match the current filters.")
    else:
        selected_pid = st.selectbox("Select Project", project_ids)
        row = filtered[filtered["project_id"] == selected_pid].iloc[0]

        st.markdown(f"### {row['project_name']} (`{row['project_id']}`)")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk Score", f"{row['risk_score']:.0f}", risk_badge(row['risk_score']))
        m2.metric("Cost Risk Prob", f"{row['cost_risk_prob']:.0f}%")
        m3.metric("Delay Risk Prob", f"{row['delay_risk_prob']:.0f}%")
        m4.metric("Physical Progress", f"{row['physical_progress_pct']:.0f}%")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Project Details**")
            details = {
                "Ministry": row["ministry"],
                "Sector": row["sector"],
                "Status": row["status"],
                "Original Cost (₹ Cr)": row["original_cost_cr"],
                "Revised Cost (₹ Cr)": row["revised_cost_cr"],
                "Expenditure (₹ Cr)": row["expenditure_cr"],
                "Cost Overrun %": row["cost_overrun_pct"],
                "Planned Duration (months)": row["planned_duration_months"],
                "Milestone Slip (months)": row["milestone_slip_months"],
                "Spend Deviation %": row["spend_deviation_pct"],
                "Start Date": row["start_date"],
                "Planned End": row["planned_end_date"],
            }
            for k, v in details.items():
                st.write(f"**{k}:** {v}")

        with c2:
            st.markdown("**Risk Indicators**")
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=row["risk_score"],
                title={"text": "Overall Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1B4F72"},
                    "steps": [
                        {"range": [0, 40], "color": "#D5F5E3"},
                        {"range": [40, 70], "color": "#FDEBD0"},
                        {"range": [70, 100], "color": "#FADBD8"},
                    ],
                    "threshold": {"line": {"color": "#E74C3C", "width": 3}, "value": 70}
                }
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3: EXPLAIN PREDICTION (SHAP)
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Explainable AI – Why is this project high/low risk?</p>', unsafe_allow_html=True)
    st.info("SHAP (SHapley Additive exPlanations) shows how each feature pushes the risk prediction higher or lower.")

    if not project_ids:
        st.warning("No projects match the current filters.")
    else:
        explain_pid = st.selectbox("Select Project to Explain", project_ids, key="explain_pid")
        explain_row = filtered[filtered["project_id"] == explain_pid].iloc[0]
        risk_type = st.radio("Explain which risk model?", ["overall_risk", "cost_risk", "delay_risk"], horizontal=True)

        if risk_type in models:
            X_row = prepare_features(pd.DataFrame([explain_row]))
            contrib = get_shap_explanation(models[risk_type], X_row, FEATURE_COLS)

            st.markdown(f"**Project:** {explain_row['project_name']} | **Predicted {risk_type.replace('_', ' ').title()} Probability:** {explain_row.get(f'{risk_type}_prob', 'N/A')}%")

            # SHAP bar chart
            fig = px.bar(
                contrib.head(9),
                x="shap_value", y="feature", orientation="h",
                color="shap_value",
                color_continuous_scale=["#27AE60", "#F7F9F9", "#E74C3C"],
                color_continuous_midpoint=0,
                labels={"shap_value": "SHAP Value (impact on risk)", "feature": "Feature"},
                title="Feature Contributions to Risk Prediction"
            )
            fig.update_layout(
                height=420, margin=dict(l=10, r=10, t=40, b=10),
                yaxis={"categoryorder": "total ascending"},
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False, font=dict(family="IBM Plex Sans")
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### Top Risk Drivers")
            top = contrib.head(5)
            for _, r in top.iterrows():
                direction = "increases risk" if r["shap_value"] > 0 else "decreases risk"
                st.write(f"• **{r['feature']}** = `{r['feature_value']:.1f}` → {direction} (SHAP: {r['shap_value']:.3f})")

# ══════════════════════════════════════════════
# TAB 4: WHAT-IF SIMULATOR
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">What-If Intervention Simulator</p>', unsafe_allow_html=True)
    st.markdown("Change key project parameters and instantly see how risk probabilities respond.")

    if not project_ids:
        st.warning("No projects match the current filters.")
    else:
        sim_pid = st.selectbox("Select Project to Simulate", project_ids, key="sim_pid")
        sim_row = filtered[filtered["project_id"] == sim_pid].iloc[0]

        st.markdown(f"**Current Risk Score:** {sim_row['risk_score']:.0f} | Cost Risk: {sim_row['cost_risk_prob']:.0f}% | Delay Risk: {sim_row['delay_risk_prob']:.0f}%")

        st.markdown("##### Adjust Parameters")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            new_progress = st.slider("Physical Progress %", 0.0, 100.0, float(sim_row["physical_progress_pct"]), 1.0)
            new_slip = st.slider("Milestone Slip (months)", 0.0, 24.0, float(sim_row["milestone_slip_months"]), 0.5)
        with col_b:
            new_spend_dev = st.slider("Spend Deviation %", -50.0, 50.0, float(sim_row["spend_deviation_pct"]), 1.0)
            new_overrun = st.slider("Cost Overrun %", 0.0, 100.0, float(sim_row["cost_overrun_pct"]), 1.0)
        with col_c:
            new_variance = st.slider("Progress Variance", 0.0, 50.0, float(sim_row["progress_variance"]), 1.0)

        if st.button("🔄 Run Simulation", type="primary"):
            changes = {
                "physical_progress_pct": new_progress,
                "milestone_slip_months": new_slip,
                "spend_deviation_pct": new_spend_dev,
                "cost_overrun_pct": new_overrun,
                "progress_variance": new_variance,
            }
            result = simulate_intervention(sim_row, changes, models)

            st.markdown("##### Simulation Results")
            r1, r2, r3 = st.columns(3)
            for target in ["overall_risk", "cost_risk", "delay_risk"]:
                orig = result["original"].get(target, 0)
                sim = result["simulated"].get(target, 0)
                delta = sim - orig
                label = target.replace("_", " ").title()
                with [r1, r2, r3][["overall_risk", "cost_risk", "delay_risk"].index(target)]:
                    st.metric(label, f"{sim:.0f}%", f"{delta:+.1f}% vs original")

            # Before/After bar
            fig = go.Figure(data=[
                go.Bar(name="Original", x=["Overall", "Cost", "Delay"],
                       y=[result["original"]["overall_risk"], result["original"]["cost_risk"], result["original"]["delay_risk"]],
                       marker_color="#5DADE2"),
                go.Bar(name="After Intervention", x=["Overall", "Cost", "Delay"],
                       y=[result["simulated"]["overall_risk"], result["simulated"]["cost_risk"], result["simulated"]["delay_risk"]],
                       marker_color="#1B4F72"),
            ])
            fig.update_layout(
                barmode="group", height=320,
                title="Risk Probability: Before vs After Intervention",
                yaxis_title="Risk Probability (%)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="IBM Plex Sans")
            )
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5: EARLY WARNING QUEUE
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">Early Warning Queue – Prioritized Intervention List</p>', unsafe_allow_html=True)
    st.markdown("Projects ranked by urgency for review. High risk score + high milestone slip + high cost overrun rise to the top.")

    queue = filtered.copy()
    queue["urgency"] = (
        queue["risk_score"] * 0.5
        + queue["milestone_slip_months"] * 3
        + queue["cost_overrun_pct"] * 0.3
        + (100 - queue["physical_progress_pct"]) * 0.15
    )
    queue = queue.sort_values("urgency", ascending=False)

    st.dataframe(
        queue[[
            "project_id", "project_name", "ministry", "sector", "status",
            "risk_score", "cost_risk_prob", "delay_risk_prob",
            "milestone_slip_months", "cost_overrun_pct", "physical_progress_pct", "urgency"
        ]].head(30).style.background_gradient(subset=["urgency", "risk_score"], cmap="RdYlGn_r"),
        use_container_width=True, height=450
    )

    st.markdown("##### Recommended Actions")
    top3 = queue.head(3)
    for i, (_, r) in enumerate(top3.iterrows(), 1):
        st.markdown(f"""
        **{i}. {r['project_name']}** (`{r['project_id']}`) – Risk Score **{r['risk_score']:.0f}**  
        → Suggested focus: {'Cost control & revised estimate review' if r['cost_risk_prob'] > r['delay_risk_prob'] else 'Schedule recovery & milestone re-baselining'}
        """)

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong>INFRAguard AI</strong> &nbsp;|&nbsp; Team Sentinel &nbsp;|&nbsp; Smart India Hackathon 2026 &nbsp;|&nbsp; PS 26103<br>
    MONITOR → PREDICT → EXPLAIN → ACT &nbsp;|&nbsp; Synthetic PAIMANA-aligned demo data
</div>
""", unsafe_allow_html=True)
