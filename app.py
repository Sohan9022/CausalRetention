import os
import sys
import textwrap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_or_create_data
from dynamic_pipeline import UniversalTLearner, auto_prepare_dataset
from optimizer import apply_financial_metrics, compare_all_strategies, optimize_retention_budget
from report_generator import generate_html_report
from uplift_model import TLearnerUpliftEngine, compute_cumulative_uplift_curve

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CausalRetention | Decision Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- TOP-TIER DESIGN SYSTEM (Linear / Supabase / Vercel style) ---
# textwrap.dedent guarantees <style> and HTML start at column 0, preventing markdown code-block parsing bugs
CUSTOM_CSS = textwrap.dedent("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Typography & Base Canvas */
html, body, [class*="css"], [class*="st-"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Global App Dark Canvas */
.stApp {
    background: radial-gradient(circle at 50% 0%, #111827 0%, #080C14 70%, #05070B 100%) !important;
    color: #F8FAFC !important;
}

/* 1. Hide Streamlit Default Top Header Clutter (Deploy button, Rainbow decoration, Hamburger) */
div[data-testid="stDecoration"] {
    display: none !important;
}
.stDeployButton {
    display: none !important;
}
[data-testid="stToolbar"] {
    display: none !important;
}
#MainMenu {
    display: none !important;
}
footer {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 1.2rem !important;
}

/* Eliminate awkward top whitespace */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 96% !important;
}

/* Top Enterprise Navbar (Linear / Vercel style) */
.enterprise-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(13, 19, 33, 0.9);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 10px 18px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
}
.nav-brand-left {
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-logo-icon {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
    width: 26px;
    height: 26px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 800;
    color: white;
}
.nav-brand-text {
    font-weight: 700;
    font-size: 14px;
    color: #F8FAFC;
    letter-spacing: -0.02em;
}
.nav-tag {
    font-size: 10px;
    font-weight: 700;
    padding: 2px 7px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #A5B4FC;
    border-radius: 4px;
    letter-spacing: 0.04em;
}
.nav-crumb {
    color: #475569;
    font-size: 13px;
}
.nav-env-name {
    color: #94A3B8;
    font-size: 12px;
    font-weight: 500;
}
.nav-telemetry-right {
    display: flex;
    align-items: center;
    gap: 14px;
}
.telemetry-item {
    font-size: 11.5px;
    color: #64748B;
    display: flex;
    align-items: center;
    gap: 5px;
}
.telemetry-val {
    color: #E2E8F0;
    font-weight: 600;
}
.telemetry-pill-green {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #34D399;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
}

/* Header Container (Frosted Glass & Linear Aesthetics) */
.brand-header-card {
    background: linear-gradient(135deg, rgba(22, 31, 48, 0.75) 0%, rgba(13, 19, 33, 0.9) 100%);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px -4px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.brand-badge {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
    box-shadow: 0 0 16px rgba(99, 102, 241, 0.4);
    color: #FFFFFF;
    font-weight: 800;
    font-size: 15px;
    letter-spacing: -0.02em;
    padding: 6px 12px;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

.live-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #34D399;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #10B981;
    border-radius: 50%;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulse-ring 2s infinite cubic-bezier(0.4, 0, 0.6, 1);
}

@keyframes pulse-ring {
    0%, 100% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    }
    50% {
        transform: scale(1.1);
        box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
    }
}

/* Environment Switcher Container */
.env-selector-wrapper {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 14px 20px 12px 20px;
    margin-bottom: 22px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}

/* Radio buttons styled like segmented pills */
div[role="radiogroup"] {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    padding-top: 4px;
}
div[role="radiogroup"] > label {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
    color: #CBD5E1 !important;
}
div[role="radiogroup"] > label:hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
    background: rgba(30, 41, 59, 0.8) !important;
    color: #FFFFFF !important;
}

/* High Contrast Metric Cards */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(20, 27, 45, 0.7) 0%, rgba(11, 16, 29, 0.8) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
    transition: transform 0.2s ease, border-color 0.2s ease !important;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(99, 102, 241, 0.3) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stMetricValue"] > div {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 28px !important;
    letter-spacing: -0.02em !important;
}
div[data-testid="stMetricLabel"] > div {
    color: #94A3B8 !important;
    font-weight: 700 !important;
    font-size: 11.5px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
div[data-testid="stMetricDelta"] > div {
    font-weight: 600 !important;
    font-size: 12px !important;
}

/* Streamlit Tabs Custom Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    color: #94A3B8 !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    padding: 10px 18px !important;
    border: none !important;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #F8FAFC !important;
    background: rgba(255, 255, 255, 0.03) !important;
}
.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
    border-bottom: 2px solid #6366F1 !important;
    background: linear-gradient(180deg, rgba(99, 102, 241, 0.08) 0%, transparent 100%) !important;
}

/* Container Card borders */
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: linear-gradient(160deg, rgba(17, 24, 39, 0.7) 0%, rgba(11, 17, 30, 0.85) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2) !important;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A0E17 0%, #060910 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}

/* Buttons */
button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* Custom quadrant archetype cards */
.dark-quad-box {
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
    border: 1px solid;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.dark-quad-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}
.dark-quad-title {
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.dark-quad-desc {
    font-size: 13px;
    line-height: 1.5;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0B0F19;
}
::-webkit-scrollbar-thumb {
    background: #1F2937;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #374151;
}
</style>
""").strip()

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_default_engine_and_data():
    """Load or train default IBM Telco Benchmark."""
    data = load_or_create_data()
    bundle_path = os.path.join("models", "uplift_model_bundle.joblib")
    if os.path.exists(bundle_path):
        engine = TLearnerUpliftEngine.load()
    else:
        from sklearn.model_selection import train_test_split

        train_df, _ = train_test_split(data, test_size=0.25, random_state=42)
        engine = TLearnerUpliftEngine(random_state=42)
        engine.fit(train_df)
        engine.save()
    scored = engine.predict_uplift(data)
    return engine, scored


@st.cache_resource
def get_bank_engine_and_data():
    """Load and train Bank Customer Churn Benchmark (10,000 Accounts)."""
    cache_path = os.path.join("data", "raw", "Bank_Churn_Modelling.csv")
    if not os.path.exists(cache_path):
        url = "https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Bank%20Churn%20Modelling.csv"
        df_raw = pd.read_csv(url)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_raw.to_csv(cache_path, index=False)
    else:
        df_raw = pd.read_csv(cache_path)

    df_prep, feat_cols = auto_prepare_dataset(df_raw, target_col="Churn", target_is_churn=True)
    engine = UniversalTLearner(random_state=42)
    engine.fit(df_prep, feat_cols)
    scored = engine.predict_uplift(df_prep)
    return engine, scored


@st.cache_resource
def get_shahid_azeem_engine_and_data():
    """Load and train Kaggle Shahid Azeem Subscription Churn Benchmark (64,374 Accounts)."""
    kaggle_paths = [
        "/kaggle/input/datasets/muhammadshahidazeem/customer-churn-dataset/customer_churn_dataset-testing-master.csv",
        "/kaggle/input/datasets/muhammadshahidazeem/customer-churn-dataset/customer_churn_dataset-training-master.csv",
        os.path.join("data", "raw", "customer_churn_dataset.csv"),
    ]
    df_raw = None
    for p in kaggle_paths:
        if os.path.exists(p):
            df_raw = pd.read_csv(p)
            break

    if df_raw is None:
        url = "https://raw.githubusercontent.com/faisalghifari17/customer-churn-prediction/main/customer_churn_dataset-testing-master.csv"
        df_raw = pd.read_csv(url)
        cache_path = os.path.join("data", "raw", "customer_churn_dataset.csv")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_raw.to_csv(cache_path, index=False)

    df_prep, feat_cols = auto_prepare_dataset(df_raw, target_col="Churn", target_is_churn=True)
    sample_train = df_prep.sample(min(15000, len(df_prep)), random_state=42).reset_index(drop=True)
    engine = UniversalTLearner(random_state=42)
    engine.fit(sample_train, feat_cols)
    scored = engine.predict_uplift(sample_train)
    return engine, scored


# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    SIDEBAR_BRAND_HTML = textwrap.dedent("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 18px; padding: 4px 0;">
        <div style="background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%); width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 800; font-size: 14px; box-shadow: 0 0 16px rgba(99, 102, 241, 0.4);">CR</div>
        <div>
            <div style="font-weight: 800; font-size: 15px; color: #FFFFFF; letter-spacing: -0.02em;">CausalRetention</div>
            <div style="font-size: 11px; color: #94A3B8;">Executive Decision System</div>
        </div>
    </div>
    """).strip()
    st.markdown(SIDEBAR_BRAND_HTML, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='font-size: 11px; font-weight: 700; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.06em;'>Financial Budget Allocation</p>", unsafe_allow_html=True)

    budget_input = st.slider(
        "Retention Budget ($)",
        min_value=2000,
        max_value=50000,
        value=15000,
        step=1000,
        help="Maximum capital allocated for retention discounts & customer outreach.",
    )

    uplift_threshold = st.slider(
        "Persuadable Cutoff (Tau)",
        min_value=0.05,
        max_value=0.30,
        value=0.10,
        step=0.01,
        help="Minimum individual treatment effect (tau) to qualify for budget allocation.",
    )

    st.markdown("---")
    st.markdown("<p style='font-size: 11px; font-weight: 700; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.06em;'>Offer Cost Parameters</p>", unsafe_allow_html=True)
    custom_cost_toggle = st.checkbox("Set Static Offer Cost", value=False)
    flat_cost = None
    if custom_cost_toggle:
        flat_cost = st.number_input("Flat Cost per Customer ($)", value=40.0, step=5.0)

    st.markdown("---")
    SIDEBAR_SPECS_HTML = textwrap.dedent("""
    <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 14px; font-size: 12px; color: #94A3B8; line-height: 1.6;">
        <div style="font-weight: 700; color: #FFFFFF; margin-bottom: 6px; font-size: 12px; display: flex; align-items: center; justify-content: space-between;">
            <span>Engine Specification</span>
            <span style="color: #10B981; font-size: 11px; font-weight: 600;">● Online</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
            <span>Architecture:</span> <strong style="color: #E2E8F0;">Two-Model T-Learner</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
            <span>Estimator:</span> <strong style="color: #E2E8F0;">XGBoost (gbtree)</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
            <span>Objective:</span> <strong style="color: #E2E8F0;">Net ROI Maximization</strong>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Optimization:</span> <strong style="color: #E2E8F0;">Greedy Knapsack</strong>
        </div>
    </div>
    """).strip()
    st.markdown(SIDEBAR_SPECS_HTML, unsafe_allow_html=True)


# ----------------- TOP ENTERPRISE NAVBAR (100% REAL & AUTHENTIC) -----------------
NAVBAR_HTML = textwrap.dedent("""
<div class="enterprise-navbar">
    <div class="nav-brand-left">
        <div class="nav-logo-icon">⚡</div>
        <span class="nav-brand-text">CausalRetention</span>
        <span class="nav-tag">Prescriptive ML</span>
        <span class="nav-crumb">/</span>
        <span class="nav-env-name">Decision Intelligence & Uplift Platform</span>
    </div>
    <div style="display: flex; align-items: center; gap: 10px;">
        <a href="https://github.com/Sohan9022/CausalRetention" target="_blank" style="text-decoration: none; color: #CBD5E1; font-size: 12px; font-weight: 600; background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.12); padding: 6px 14px; border-radius: 8px; transition: all 0.2s ease;">
            <span style="color: #818CF8;">★</span> GitHub Repo ↗
        </a>
    </div>
</div>
""").strip()
st.markdown(NAVBAR_HTML, unsafe_allow_html=True)


# ----------------- TOP HEADER BANNER (NO RAW TEXT) -----------------
HEADER_HTML = textwrap.dedent("""
<div class="brand-header-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div class="brand-badge">CR</div>
            <div>
                <h1 style="color: #FFFFFF; font-size: 22px; font-weight: 800; margin: 0; letter-spacing: -0.02em;">
                    CausalRetention <span style="font-size: 13.5px; font-weight: 600; color: #818CF8; margin-left: 6px; letter-spacing: 0;">Portfolio Decision Intelligence</span>
                </h1>
                <div style="color: #94A3B8; font-size: 13px; margin-top: 4px;">
                    Prescriptive CATE Uplift Engine &middot; Counterfactual Intervention &middot; Knapsack Budget Optimizer
                </div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div class="live-indicator">
                <span class="pulse-dot"></span>
                <span>Production Engine Active &middot; XGBoost T-Learner v2.4</span>
            </div>
        </div>
    </div>
</div>
""").strip()
st.markdown(HEADER_HTML, unsafe_allow_html=True)


# ----------------- PROMINENT ENVIRONMENT SWITCHER -----------------
ENV_SWITCH_HTML = textwrap.dedent("""
<div style="margin-bottom: 8px;">
    <span style="font-size: 11.5px; font-weight: 700; color: #94A3B8; letter-spacing: 0.06em; text-transform: uppercase;">
        ⚡ Enterprise Data Environment Cohort
    </span>
</div>
""").strip()
st.markdown(ENV_SWITCH_HTML, unsafe_allow_html=True)

dataset_source = st.radio(
    "Active Environment:",
    [
        "📊 Telecom Retention Benchmark (IBM Telco - 7,043 Accounts)",
        "🏦 Banking Retention Benchmark (Bank Churn - 10,000 Accounts)",
        "⚡ Kaggle Subscription Benchmark (Shahid Azeem - 64,374 Accounts)",
        "📁 Upload Custom Dataset (Any CSV File)",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

st.write("")

# ----------------- DATASET LOGIC -----------------
active_engine = None
active_scored_df = None
current_dataset_name = "IBM Telco Retention Cohort (7,043 Accounts)"

if dataset_source.startswith("📊 Telecom"):
    active_engine, active_scored_df = get_default_engine_and_data()
    current_dataset_name = "IBM Telco Retention Benchmark (7,043 Accounts)"

elif dataset_source.startswith("🏦 Banking"):
    with st.spinner("Loading Bank Churn Benchmark & scoring uplift (10,000 Accounts)..."):
        active_engine, active_scored_df = get_bank_engine_and_data()
        current_dataset_name = "Bank Customer Churn Benchmark (10,000 Accounts)"

elif dataset_source.startswith("⚡ Kaggle"):
    with st.spinner("Loading Kaggle Shahid Azeem Churn Benchmark (64,374 Accounts)..."):
        active_engine, active_scored_df = get_shahid_azeem_engine_and_data()
        current_dataset_name = "Kaggle Subscription Churn Benchmark (Shahid Azeem - 64,374 Accounts)"

else:
    # Custom Dataset Upload Workflow Card
    with st.container(border=True):
        st.markdown("### 📁 Custom Dataset Upload & Auto-Mapper")
        st.write("Upload any tabular churn or customer retention CSV dataset. The engine will auto-detect features, train the T-Learner uplift model, and calibrate financial returns.")

        up_col1, up_col2 = st.columns([1, 1])

        with up_col1:
            uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="custom_csv_uploader")

        if uploaded_file is not None:
            try:
                try:
                    raw_user_df = pd.read_csv(uploaded_file)
                    if len(raw_user_df.columns) == 1 and ";" in str(raw_user_df.columns[0]):
                        uploaded_file.seek(0)
                        raw_user_df = pd.read_csv(uploaded_file, sep=";")
                except Exception:
                    uploaded_file.seek(0)
                    raw_user_df = pd.read_csv(uploaded_file, sep=None, engine="python")

                st.success(f"✓ Loaded `{uploaded_file.name}` ({len(raw_user_df):,} rows, {len(raw_user_df.columns)} columns)")

                all_cols = list(raw_user_df.columns)
                target_candidates = [c for c in all_cols if any(k in c.lower() for k in ["churn", "retain", "leave", "cancel", "exited", "attrit"])]
                default_target_idx = all_cols.index(target_candidates[0]) if target_candidates else 0

                id_candidates = [c for c in all_cols if any(k in c.lower() for k in ["customerid", "customer_id", "cust_id", "custid", "clientnum", "client_id", "accountid", "user_id", "id"])]
                default_id_idx = (all_cols.index(id_candidates[0]) + 1) if id_candidates else 0

                with up_col2:
                    st.markdown("#### Schema Mapping Wizard")
                    m_row1, m_row2 = st.columns(2)
                    with m_row1:
                        target_col = st.selectbox("Target Column (Outcome):", all_cols, index=default_target_idx)
                        target_is_churn = st.radio("Target Meaning:", ["1 = Churn (0 = Retained)", "1 = Retained (0 = Churn)"], index=0) == "1 = Churn (0 = Retained)"

                        id_options = ["Auto-Detect / Generate"] + all_cols
                        id_col_sel = st.selectbox("Customer ID Column (Optional):", id_options, index=default_id_idx)
                        actual_id_col = None if id_col_sel.startswith("Auto-Detect") else id_col_sel

                    with m_row2:
                        treatment_candidates = ["None (Simulate 50/50 A/B Trial)"] + all_cols
                        treatment_col_sel = st.selectbox("Treatment Column (Optional):", treatment_candidates, index=0)
                        actual_treat_col = None if treatment_col_sel.startswith("None") else treatment_col_sel

                        clv_candidates = ["Auto-Calibrate from Spend/Balance"] + all_cols
                        clv_col_sel = st.selectbox("CLV / Spend Column (Optional):", clv_candidates, index=0)
                        actual_clv_col = None if clv_col_sel.startswith("Auto-Calibrate") else clv_col_sel

                    if st.button("⚡ Train Uplift Model on Uploaded CSV", type="primary", use_container_width=True):
                        with st.spinner("Processing schema, imputing, encoding, and fitting Two-Model XGBoost..."):
                            clean_df, feat_cols = auto_prepare_dataset(
                                df=raw_user_df,
                                target_col=target_col,
                                target_is_churn=target_is_churn,
                                treatment_col=actual_treat_col,
                                id_col=actual_id_col,
                                clv_col=actual_clv_col,
                            )
                            uni_engine = UniversalTLearner(random_state=42)
                            uni_engine.fit(clean_df, feat_cols)
                            scored_custom = uni_engine.predict_uplift(clean_df)
                            st.session_state["custom_scored"] = scored_custom
                            st.session_state["custom_engine"] = uni_engine
                            st.session_state["custom_name"] = uploaded_file.name
                            st.success("✓ Training complete! All tabs updated below.")

            except Exception as e:
                st.error(f"Error parsing uploaded CSV: {e}")

        if "custom_scored" in st.session_state:
            active_scored_df = st.session_state["custom_scored"]
            active_engine = st.session_state["custom_engine"]
            current_dataset_name = f"Custom: {st.session_state.get('custom_name', 'File')} ({len(active_scored_df):,} Accounts)"
        else:
            st.info("👆 Upload your CSV file above and click '⚡ Train Uplift Model' to populate the dashboard.")
            st.stop()


# ----------------- FINANCIAL COMPUTATION -----------------
scored_fin = apply_financial_metrics(
    active_scored_df, default_offer_cost=flat_cost, uplift_threshold=uplift_threshold
)
opt_results = optimize_retention_budget(
    scored_fin, budget_limit=budget_input, strategy="causal_uplift"
)
summary = opt_results["summary"]
targeted_df = opt_results["targeted_df"]
comp_df = compare_all_strategies(scored_fin, budget_limit=budget_input)


# ----------------- TOP EXECUTIVE KPI ROW (ENTERPRISE HIGH-CONTRAST) -----------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    KPI1_HTML = textwrap.dedent(f"""
    <div style="background: linear-gradient(145deg, rgba(20, 27, 45, 0.7) 0%, rgba(11, 16, 29, 0.8) 100%); border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid #6366F1; border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);">
        <div style="color: #94A3B8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">Capital Allocated</div>
        <div style="color: #FFFFFF; font-size: 26px; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 4px;">${summary['budget_spent']:,.0f}</div>
        <div style="color: #818CF8; font-size: 12px; font-weight: 600;">${summary['budget_remaining']:,.0f} unspent remaining</div>
    </div>
    """).strip()
    st.markdown(KPI1_HTML, unsafe_allow_html=True)

with k2:
    KPI2_HTML = textwrap.dedent(f"""
    <div style="background: linear-gradient(145deg, rgba(20, 27, 45, 0.7) 0%, rgba(11, 16, 29, 0.8) 100%); border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid #10B981; border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);">
        <div style="color: #94A3B8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">Customers Saved</div>
        <div style="color: #FFFFFF; font-size: 26px; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 4px;">+{summary['expected_retained_lift']:.0f}</div>
        <div style="color: #34D399; font-size: 12px; font-weight: 600;">{summary['customers_targeted']} accounts targeted</div>
    </div>
    """).strip()
    st.markdown(KPI2_HTML, unsafe_allow_html=True)

with k3:
    KPI3_HTML = textwrap.dedent(f"""
    <div style="background: linear-gradient(145deg, rgba(20, 27, 45, 0.7) 0%, rgba(11, 16, 29, 0.8) 100%); border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid #06B6D4; border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);">
        <div style="color: #94A3B8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">Net Preserved Revenue</div>
        <div style="color: #FFFFFF; font-size: 26px; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 4px;">+${summary['net_profit']:,.0f}</div>
        <div style="color: #22D3EE; font-size: 12px; font-weight: 600;">${summary['gross_revenue_saved']:,.0f} gross saved</div>
    </div>
    """).strip()
    st.markdown(KPI3_HTML, unsafe_allow_html=True)

with k4:
    KPI4_HTML = textwrap.dedent(f"""
    <div style="background: linear-gradient(145deg, rgba(20, 27, 45, 0.7) 0%, rgba(11, 16, 29, 0.8) 100%); border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid #F59E0B; border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);">
        <div style="color: #94A3B8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">Campaign Net ROI</div>
        <div style="color: #FFFFFF; font-size: 26px; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 4px;">+{summary['roi_percent']:.1f}%</div>
        <div style="color: #FBBF24; font-size: 12px; font-weight: 600;">Profit / Capital Spent</div>
    </div>
    """).strip()
    st.markdown(KPI4_HTML, unsafe_allow_html=True)

st.write("")

# ----------------- WORKSPACE TABS -----------------
tab_bench, tab_cohorts, tab_crm, tab_report, tab_sim = st.tabs(
    [
        "📊 Strategy Comparison",
        "🧩 Behavioral Cohorts",
        "📋 CRM Target Roster",
        "📄 Executive Brief",
        "🧪 Customer Simulator",
    ]
)

# ----------------- TAB 1: STRATEGY COMPARISON -----------------
with tab_bench:
    with st.container(border=True):
        st.subheader("1. Strategic Targeting Comparison")
        st.caption(
            f"Comparing performance of three targeting paradigms under the identical **${budget_input:,.2f}** budget allocation:"
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.subheader("Net Profit Generated ($)")
            causal_p = optimize_retention_budget(scored_fin, budget_input, "causal_uplift")["summary"]["net_profit"]
            churn_p = optimize_retention_budget(scored_fin, budget_input, "traditional_churn")["summary"]["net_profit"]
            random_p = optimize_retention_budget(scored_fin, budget_input, "random")["summary"]["net_profit"]

            fig_p = go.Figure()
            fig_p.add_trace(
                go.Bar(
                    x=["Causal Uplift (Ours)", "Standard Churn ML", "Random Baseline"],
                    y=[causal_p, churn_p, random_p],
                    marker_color=["#6366F1", "#F43F5E", "#64748B"],
                    text=[f"${causal_p:,.0f}", f"${churn_p:,.0f}", f"${random_p:,.0f}"],
                    textposition="auto",
                    textfont=dict(color="#FFFFFF", size=13),
                )
            )
            fig_p.update_layout(
                plot_bgcolor="#111827",
                paper_bgcolor="#111827",
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(gridcolor="#1F2937", zerolinecolor="#475569", tickfont=dict(color="#94A3B8")),
                xaxis=dict(tickfont=dict(color="#F8FAFC", size=13)),
            )
            st.plotly_chart(fig_p, use_container_width=True)

    with c2:
        with st.container(border=True):
            st.subheader("Cumulative Uplift (Qini Curve)")
            curve_data = compute_cumulative_uplift_curve(scored_fin, score_col="uplift_score")
            fig_q = go.Figure()
            fig_q.add_trace(
                go.Scatter(
                    x=curve_data["percent_targeted"],
                    y=curve_data["cumulative_incremental_retained"],
                    mode="lines+markers",
                    name="Causal Uplift",
                    line=dict(color="#6366F1", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(99, 102, 241, 0.15)",
                )
            )
            max_lift = curve_data["cumulative_incremental_retained"].iloc[-1]
            fig_q.add_trace(
                go.Scatter(
                    x=[0, 100],
                    y=[0, max_lift],
                    mode="lines",
                    name="Random Baseline",
                    line=dict(color="#64748B", dash="dash"),
                )
            )
            fig_q.update_layout(
                plot_bgcolor="#111827",
                paper_bgcolor="#111827",
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(title="Customer Base Targeted (%)", gridcolor="#1F2937", tickfont=dict(color="#94A3B8"), title_font=dict(color="#94A3B8")),
                yaxis=dict(title="Incremental Saved", gridcolor="#1F2937", tickfont=dict(color="#94A3B8"), title_font=dict(color="#94A3B8")),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#F8FAFC")),
            )
            st.plotly_chart(fig_q, use_container_width=True)

# ----------------- TAB 2: BEHAVIORAL COHORTS -----------------
with tab_cohorts:
    with st.container(border=True):
        st.subheader("2. The 4 Behavioral Uplift Quadrants")
        st.caption("Cross-referencing baseline retention probability against causal intervention uplift:")

        col_q1, col_q2 = st.columns([1, 2])

        q_counts = scored_fin["quadrant"].value_counts().reset_index()
        q_counts.columns = ["Quadrant", "Count"]

        with col_q1:
            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=q_counts["Quadrant"],
                        values=q_counts["Count"],
                        hole=0.6,
                        marker=dict(colors=["#6366F1", "#10B981", "#F59E0B", "#F43F5E"]),
                    )
                ]
            )
            fig_donut.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#111827",
                showlegend=False,
                font=dict(color="#F8FAFC"),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_q2:
            scatter_sample = scored_fin.sample(min(600, len(scored_fin)), random_state=42)
            fig_sc = px.scatter(
                scatter_sample,
                x="baseline_churn_prob",
                y="uplift_score",
                color="quadrant",
                color_discrete_map={
                    "Persuadable (High Priority)": "#6366F1",
                    "Sure Thing (Organic Retain)": "#10B981",
                    "Lost Cause (Unresponsive)": "#F59E0B",
                    "Sleeping Dog (Do Not Contact)": "#F43F5E",
                },
                labels={
                    "baseline_churn_prob": "Baseline Churn Risk (Traditional ML)",
                    "uplift_score": "Causal Uplift Score (Tau)",
                    "quadrant": "Quadrant",
                },
            )
            fig_sc.update_layout(
                plot_bgcolor="#111827",
                paper_bgcolor="#111827",
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(gridcolor="#1F2937", tickfont=dict(color="#94A3B8"), title_font=dict(color="#94A3B8")),
                yaxis=dict(gridcolor="#1F2937", tickfont=dict(color="#94A3B8"), title_font=dict(color="#94A3B8")),
                legend=dict(font=dict(color="#F8FAFC")),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("#### Behavioral Archetype Decision Matrix:")
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.markdown(
            textwrap.dedent("""
            <div class="dark-quad-box" style="background-color: rgba(99, 102, 241, 0.1); border-color: rgba(99, 102, 241, 0.4);">
                <div class="dark-quad-title" style="color: #A5B4FC;">🎯 Persuadables</div>
                <div class="dark-quad-desc" style="color: #E0E7FF;">High uplift (&tau; &ge; +0.10). Churn without offer, but stay with one. <strong style="color: #818CF8;">Concentrate budget here.</strong></div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    with p2:
        st.markdown(
            textwrap.dedent("""
            <div class="dark-quad-box" style="background-color: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.4);">
                <div class="dark-quad-title" style="color: #6EE7B7;">🛡️ Sure Things</div>
                <div class="dark-quad-desc" style="color: #D1FAE5;">High organic retention (>90%). Discounts create deadweight revenue loss. <strong style="color: #34D399;">Suppress offers.</strong></div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    with p3:
        st.markdown(
            textwrap.dedent("""
            <div class="dark-quad-box" style="background-color: rgba(245, 158, 11, 0.1); border-color: rgba(245, 158, 11, 0.4);">
                <div class="dark-quad-title" style="color: #FCD34D;">⚠️ Lost Causes</div>
                <div class="dark-quad-desc" style="color: #FEF3C7;">Severe churn risk, near-zero uplift. Leave regardless of incentive. <strong style="color: #FBBF24;">Suppress to save spend.</strong></div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    with p4:
        st.markdown(
            textwrap.dedent("""
            <div class="dark-quad-box" style="background-color: rgba(244, 63, 94, 0.1); border-color: rgba(244, 63, 94, 0.4);">
                <div class="dark-quad-title" style="color: #FDA4AF;">🛑 Sleeping Dogs</div>
                <div class="dark-quad-desc" style="color: #FFE4E6;">Negative uplift (&tau; &lt; 0). Reaching out triggers cancellations. <strong style="color: #FB7185;">Blacklist from campaigns.</strong></div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

# ----------------- TAB 3: CRM TARGET ROSTER -----------------
with tab_crm:
    with st.container(border=True):
        st.subheader("3. Actionable CRM Campaign Dispatch Roster")
        st.caption(
            f"Selected **{len(targeted_df)}** optimal customer accounts prioritized by Expected Net Profit under the **${budget_input:,.2f}** budget limit:"
        )

        display_cols = [
            "customerID",
            "uplift_score",
            "campaign_offer_cost",
            "expected_gross_revenue_saved",
            "expected_net_profit",
            "quadrant",
        ]
        for col in ["Contract", "tenure", "MonthlyCharges"]:
            if col in targeted_df.columns:
                display_cols.insert(1, col)

        df_show = targeted_df[display_cols].sort_values(by="expected_net_profit", ascending=False).copy()
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        csv_bytes = df_show.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Campaign Target List (CSV)",
            data=csv_bytes,
            file_name="campaign_target_roster.csv",
            mime="text/csv",
        )

# ----------------- TAB 4: EXECUTIVE REPORT -----------------
with tab_report:
    with st.container(border=True):
        st.subheader("4. Board-Ready Executive Brief")
        st.caption("Auto-generated, printable report prepared for C-suite presentation and operational review.")

        c_r1, c_r2 = st.columns([2, 1])

        with c_r2:
            if st.button("🔄 Refresh Executive Report", use_container_width=True):
                generate_html_report(
                    summary=summary,
                    comparison_df=comp_df,
                    targeted_df=targeted_df,
                    output_path="reports/executive_retention_report.html",
                )
                st.success("Executive brief refreshed successfully!")

        report_file_path = "reports/executive_retention_report.html"
        if os.path.exists(report_file_path):
            with open(report_file_path, "r", encoding="utf-8") as f:
                html_body = f.read()

            with c_r1:
                st.download_button(
                    label="📄 Download Executive Brief (HTML / Print to PDF)",
                    data=html_body,
                    file_name="Executive_Retention_Brief.html",
                    mime="text/html",
                    use_container_width=True,
                )

            st.components.v1.html(html_body, height=650, scrolling=True)
        else:
            st.info("Click 'Refresh Executive Report' above to render.")

# ----------------- TAB 5: WHAT-IF SIMULATOR -----------------
with tab_sim:
    with st.container(border=True):
        st.subheader("5. Interactive Customer Live Scorer")
        st.caption("Simulate any account profile to inspect live causal uplift and policy recommendation:")

        if isinstance(active_engine, TLearnerUpliftEngine):
            sim1, sim2 = st.columns(2)

            with sim1:
                st.markdown("##### Account Parameters")
                sim_tenure = st.slider("Tenure (Months)", 1, 72, 8)
                sim_monthly = st.slider("Monthly Charges / Spend ($)", 15.0, 150.0, 75.0)
                sim_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                sim_support = st.selectbox("Tech Support Active", ["No", "Yes", "No internet service"])

            with sim2:
                sim_row = pd.DataFrame(
                    [
                        {
                            "customerID": "SIM-01",
                            "gender": "Female",
                            "SeniorCitizen": "No",
                            "Partner": "No",
                            "Dependents": "No",
                            "tenure": sim_tenure,
                            "PhoneService": "Yes",
                            "MultipleLines": "No",
                            "InternetService": "Fiber optic",
                            "OnlineSecurity": "No",
                            "OnlineBackup": "No",
                            "DeviceProtection": "No",
                            "TechSupport": sim_support,
                            "StreamingTV": "No",
                            "StreamingMovies": "No",
                            "Contract": sim_contract,
                            "PaperlessBilling": "Yes",
                            "PaymentMethod": "Electronic check",
                            "MonthlyCharges": sim_monthly,
                            "TotalCharges": sim_monthly * sim_tenure,
                            "annual_clv": sim_monthly * 12,
                            "offer_cost": round(15.0 + (sim_monthly * 0.15 * 3), 2),
                        }
                    ]
                )

                try:
                    sim_res = active_engine.predict_uplift(sim_row)
                    sim_scored = apply_financial_metrics(sim_res)

                    tau_val = sim_scored.iloc[0]["uplift_score"]
                    p0_val = sim_scored.iloc[0]["p_retained_control"]
                    p1_val = sim_scored.iloc[0]["p_retained_treatment"]
                    quad_val = sim_scored.iloc[0]["quadrant"]
                    net_val = sim_scored.iloc[0]["expected_net_profit"]

                    st.markdown("##### Model Inference")
                    with st.container(border=True):
                        st.metric("Individual Treatment Effect (Tau)", f"{tau_val:+.2f}")

                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("P(Stay | No Offer)", f"{p0_val * 100:.1f}%")
                        with m2:
                            st.metric("P(Stay | Offer)", f"{p1_val * 100:.1f}%")

                        st.markdown(f"**Classification:** `{quad_val}`")
                        st.markdown(f"**Expected Net Profit:** `${net_val:,.2f}`")

                except Exception as e:
                    st.error(f"Simulator error: {e}")
        else:
            # Universal Dynamic Simulator for Banking & Custom Datasets
            sim_col1, sim_col2 = st.columns([1, 1])

            sample_ids = active_scored_df["customerID"].head(200).tolist()
            with sim_col1:
                st.markdown("##### Account Selection & Profile")
                selected_id = st.selectbox("Choose Customer Account ID to Inspect:", sample_ids)
                cust_match = active_scored_df[active_scored_df["customerID"] == selected_id].iloc[0]

                # Show key feature values
                feat_dict = {}
                feature_list = getattr(active_engine, "feature_cols", [c for c in active_scored_df.columns if c not in ["customerID", "retained", "churn", "treatment", "annual_clv", "offer_cost", "uplift_score", "quadrant"]])
                for f in feature_list[:8]:
                    feat_dict[f] = str(cust_match[f])
                st.json(feat_dict)

            with sim_col2:
                st.markdown("##### Causal Model Inference & ROI")
                tau_val = float(cust_match["uplift_score"])
                p0_val = float(cust_match["p_retained_control"])
                p1_val = float(cust_match["p_retained_treatment"])
                quad_val = str(cust_match["quadrant"])
                clv_val = float(cust_match.get("campaign_clv", cust_match.get("annual_clv", 600.0)))
                cost_val = float(cust_match.get("campaign_offer_cost", cust_match.get("offer_cost", 40.0)))
                net_val = float(cust_match.get("expected_net_profit", round((tau_val * clv_val) - cost_val, 2)))

                with st.container(border=True):
                    st.metric("Individual Treatment Effect (Tau)", f"{tau_val:+.2f}")

                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("P(Stay | No Offer)", f"{p0_val * 100:.1f}%")
                    with m2:
                        st.metric("P(Stay | Offer)", f"{p1_val * 100:.1f}%")

                    st.markdown(f"**Classification:** `{quad_val}`")
                    st.markdown(f"**Annual Customer Value (CLV):** `${clv_val:,.2f}`")
                    st.markdown(f"**Campaign Offer Cost:** `${cost_val:,.2f}`")
                    st.markdown(f"**Expected Net Profit:** `${net_val:,.2f}`")
