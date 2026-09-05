"""
ConfigPilot — AI Configuration & Reliability Copilot
Built for the SandDisk Challenge
Validated Engineering Prototype with Leakage-Audited Inputs
"""

import os
import sys

# Ensure src directory is in sys.path for module resolution
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Import backend decision engine & ingestion modules from src
from chatbot import process_sanddisk_query
from explanation import (
    EXPLANATION_FEATURES,
    calculate_local_sensitivity,
    get_feature_percentiles,
    get_global_feature_importances,
)
from ingestion import load_execution_data, validate_dataset_schema
from leakage_audit import run_feature_leakage_audit
from prediction import predict_failure_risk
from preprocessing import CONFIG_FEATURES, prepare_feature_matrix
from recommender import recommend_safe_configurations
from similarity import find_similar_configurations
from what_if import multi_parameter_what_if, single_parameter_what_if

# ==============================================================================
# 1. Page Configuration & Linear/Vercel Design System
# ==============================================================================
st.set_page_config(
    page_title="ConfigPilot — Reliability Engineering Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS: Slate palette (#0b0f19 canvas, single #2563eb accent, Inter typeface)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ── Global Canvas ─────────────────────────────────────────────── */
    .stApp {
        background: #0b0f19;
        color: #f9fafb;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* ── Landing: balanced top bar ─────────────────────────────────── */
    .landing-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.9rem 0 0.5rem 0;
        border-bottom: 1px solid #1a2030;
        margin-bottom: 0;
    }
    .landing-wordmark {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.82rem;
        font-weight: 700;
        color: #f9fafb;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .landing-wordmark-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #2563eb;
        display: inline-block;
    }
    .landing-topbar-right {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.78rem;
        color: #6b7280;
    }
    .landing-deploy-pill {
        background: #1e3a8a;
        color: #93c5fd;
        border: 1px solid #2563eb44;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .landing-kebab {
        color: #4b5563;
        font-size: 1rem;
        cursor: default;
    }

    /* ── Landing: main composition block (upper-40%) ───────────────── */
    .landing-stage {
        max-width: 560px;
        /* push to upper-40% of viewport: ~8vh top, let content height anchor it */
        margin: 8vh auto 0 auto;
        text-align: center;
        position: relative;
    }

    /* Subtle radial glow focused on the headline — not the whole page */
    .landing-stage::before {
        content: '';
        position: absolute;
        top: -60px;
        left: 50%;
        transform: translateX(-50%);
        width: 480px;
        height: 260px;
        background: radial-gradient(ellipse at center, rgba(37,99,235,0.18) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }

    /* ── Tagline eyebrow ───────────────────────────────────────────── */
    .hero-eyebrow {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #3b82f6;
        margin-bottom: 0.6rem;
        position: relative;
        z-index: 1;
    }

    /* ── Big headline — the type IS the design ─────────────────────── */
    .hero-headline {
        font-size: clamp(3rem, 6vw, 4rem);
        font-weight: 900;
        color: #f9fafb;
        letter-spacing: -0.04em;
        line-height: 1.0;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    .hero-headline-accent {
        color: #2563eb;
    }

    /* ── One-line description ──────────────────────────────────────── */
    .hero-desc {
        font-size: 0.96rem;
        color: #94a3b8;
        line-height: 1.6;
        margin-bottom: 2rem;
        position: relative;
        z-index: 1;
    }

    /* ── Inline risk-scale gauge (no box, lives on bare canvas) ────── */
    .gauge-wrap {
        margin: 0 auto 1.5rem auto;
        position: relative;
        z-index: 1;
    }
    .gauge-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.45rem;
    }
    .gauge-label-left  { font-size: 0.7rem; color: #059669; font-weight: 600; }
    .gauge-label-center {
        font-size: 0.72rem;
        color: #60a5fa;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .gauge-label-right { font-size: 0.7rem; color: #ef4444; font-weight: 600; }
    .gauge-track {
        height: 6px;
        border-radius: 3px;
        background: #1e293b;
        display: flex;
        overflow: visible;
        position: relative;
    }
    .gauge-seg-low {
        width: 29%;
        height: 100%;
        background: linear-gradient(90deg, #064e3b, #059669);
        border-radius: 3px 0 0 3px;
    }
    .gauge-seg-high {
        flex: 1;
        height: 100%;
        background: linear-gradient(90deg, #7f1d1d, #450a0a);
        border-radius: 0 3px 3px 0;
    }
    .gauge-pin {
        position: absolute;
        left: 29%;
        top: -5px;
        width: 2px;
        height: 16px;
        background: #60a5fa;
        border-radius: 1px;
        box-shadow: 0 0 8px rgba(96,165,250,0.8);
        transform: translateX(-50%);
    }
    .gauge-caption {
        font-size: 0.7rem;
        color: #475569;
        margin-top: 0.5rem;
        text-align: center;
        letter-spacing: 0.01em;
    }

    /* ── Typography Scale (workspace) ─────────────────────────────── */
    .title-primary {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f9fafb;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .subtitle-muted {
        font-size: 0.875rem;
        font-weight: 400;
        color: #9ca3af;
        margin-bottom: 1.25rem;
    }

    /* ── Status Bar (workspace) ────────────────────────────────────── */
    .status-bar {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 6px;
        padding: 0.55rem 1rem;
        font-size: 0.82rem;
        color: #d1d5db;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.25rem;
    }

    /* ── Sidebar ───────────────────────────────────────────────────── */
    div[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1f2937 !important;
    }
    .nav-group-header {
        font-size: 0.75rem;
        font-weight: 500;
        color: #6b7280;
        margin-top: 1.1rem;
        margin-bottom: 0.25rem;
        padding-left: 0.4rem;
        text-transform: capitalize;
    }
    .nav-group-space { margin-bottom: 0.4rem; }
    div[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent !important;
        border: none !important;
        color: #9ca3af !important;
        text-align: left !important;
        justify-content: flex-start !important;
        font-size: 0.84rem !important;
        padding: 0.35rem 0.6rem !important;
        margin-bottom: 0.1rem !important;
        font-weight: 400 !important;
        border-radius: 4px !important;
        box-shadow: none !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #1f2937 !important;
        color: #f9fafb !important;
    }
    div[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #1e3a8a !important;
        border: none !important;
        border-left: 3px solid #2563eb !important;
        color: #f9fafb !important;
        text-align: left !important;
        justify-content: flex-start !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        padding: 0.35rem 0.6rem !important;
        margin-bottom: 0.1rem !important;
        border-radius: 4px !important;
        box-shadow: none !important;
    }

    /* ── Primary Accent Button ─────────────────────────────────────── */
    div[data-testid="stAppViewContainer"] button[kind="primary"] {
        background-color: #2563eb !important;
        border: 1px solid #2563eb !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
    }
    div[data-testid="stAppViewContainer"] button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
    }

    /* ── Panels & Callouts ─────────────────────────────────────────── */
    .panel-box {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .info-callout {
        background-color: #111827;
        border-left: 3px solid #2563eb;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #d1d5db;
        margin: 1rem 0;
        line-height: 1.5;
    }

    /* ── Status Badges ─────────────────────────────────────────────── */
    .badge-supported   { background:#064e3b; color:#a7f3d0; padding:2px 8px; border-radius:4px; font-size:0.78rem; font-weight:600; }
    .badge-partial     { background:#713f12; color:#fef08a; padding:2px 8px; border-radius:4px; font-size:0.78rem; font-weight:600; }
    .badge-incompatible{ background:#7f1d1d; color:#fecaca; padding:2px 8px; border-radius:4px; font-size:0.78rem; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. Session State Initialization
# ==============================================================================
if "landing_passed" not in st.session_state:
    st.session_state.landing_passed = False

if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard Overview"

if "active_workspace" not in st.session_state:
    st.session_state.active_workspace = "Session Workspace | Local Browser Scope"

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "synthetic_execution_logs_10000.csv (4,939 Runs)"

if "custom_df" not in st.session_state:
    st.session_state.custom_df = None

if "validation_report" not in st.session_state:
    st.session_state.validation_report = None

if "active_config" not in st.session_state:
    st.session_state.active_config = {
        "workload_type": "MIXED_IO",
        "traffic_intensity": "MEDIUM",
        "cache_size_mb": 512,
        "cache_policy": "ADAPTIVE",
        "memory_allocation_mb": 4096,
        "memory_policy": "DYNAMIC",
        "thread_count": 32,
        "cpu_cores": 8,
        "queue_depth": 32,
        "io_parallelism": 32,
        "request_size_kb": 64,
        "io_scheduler": "DYNAMIC",
        "read_write_ratio": 50.0,
    }


# ==============================================================================
# 3. Landing / Launch Screen View (Refined Hero & Gauge Composition)
# ==============================================================================
if not st.session_state.landing_passed:
    # ── Balanced top bar: wordmark left, deploy pill + kebab right ──
    st.markdown(
        """
        <div class='landing-topbar'>
            <div class='landing-wordmark'>
                <span class='landing-wordmark-dot'></span>
                ConfigPilot
            </div>
            <div class='landing-topbar-right'>
                <span class='landing-deploy-pill'>Prototype Build</span>
                <span class='landing-kebab'>⋯</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Centered layout: one narrow column for hero composition ─────
    _, col_hero, _ = st.columns([1, 2, 1])
    with col_hero:
        hero_html = (
            "<div class='landing-stage'>"
            "<div class='hero-eyebrow'>AI Configuration &amp; Reliability Copilot</div>"
            "<div class='hero-headline'>Config<span class='hero-headline-accent'>Pilot</span></div>"
            "<div class='hero-desc'>Evaluates 13 controllable system configuration parameters "
            "to estimate failure risk before workload execution.</div>"
            "<div class='gauge-wrap'>"
            "<div class='gauge-header'>"
            "<span class='gauge-label-left'>&#9679; LOW RISK</span>"
            "<span class='gauge-label-center'>&#11045; 29.00% THRESHOLD</span>"
            "<span class='gauge-label-right'>&#9679; HIGH RISK</span>"
            "</div>"
            "<div class='gauge-track'>"
            "<div class='gauge-seg-low'></div>"
            "<div class='gauge-seg-high'></div>"
            "<div class='gauge-pin'></div>"
            "</div>"
            "<div class='gauge-caption'>"
            "Decision boundary &middot; 29.00% failure probability (Validation-set F&#8321; optimised)"
            "</div>"
            "</div>"
            "</div>"
        )
        st.markdown(hero_html, unsafe_allow_html=True)

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("Launch ConfigPilot", use_container_width=True, type="primary", key="landing_cta"):
            st.session_state.landing_passed = True
            st.rerun()

    st.stop()


# ==============================================================================
# 4. Artifact Loading & Caching
# ==============================================================================
@st.cache_resource
def load_configpilot_artifacts():
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, "models", "sanddisk_failure_model.pkl")
    threshold_path = os.path.join(project_root, "models", "sanddisk_threshold.pkl")
    data_path = os.path.join(project_root, "data", "synthetic_execution_logs_10000.csv")

    if not os.path.exists(model_path) or not os.path.exists(threshold_path):
        st.error("Model artifacts not found! Please check model directory.")
        st.stop()

    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}!")
        st.stop()

    model = joblib.load(model_path)
    threshold = float(joblib.load(threshold_path))
    df_raw = pd.read_csv(data_path)
    X_matrix, _, _, _ = prepare_feature_matrix(df_raw)

    return model, threshold, df_raw, X_matrix


# Load artifacts
try:
    model, best_threshold, df_default, X_default = load_configpilot_artifacts()
except Exception as e:
    st.error(f"Error loading ConfigPilot artifacts: {e}")
    st.stop()

# Active Dataset Selection
if st.session_state.custom_df is not None:
    df_historical = st.session_state.custom_df
    try:
        X_historical, _, _, _ = prepare_feature_matrix(df_historical)
    except Exception:
        X_historical = X_default
else:
    df_historical = df_default
    X_historical = X_default


def get_active_config_row(baseline_df: pd.DataFrame, config_dict: dict) -> pd.DataFrame:
    row = baseline_df.iloc[[0]].copy()
    for k, v in config_dict.items():
        if k in row.columns:
            row.at[row.index[0], k] = v
    return row


# ==============================================================================
# 5. Header & Structural Nested Sidebar Navigation
# ==============================================================================
current_compat = (
    st.session_state.validation_report["compatibility_status"]
    if st.session_state.validation_report
    else "SUPPORTED"
)
badge_class = (
    "badge-supported" if current_compat == "SUPPORTED"
    else ("badge-partial" if current_compat == "PARTIALLY COMPATIBLE" else "badge-incompatible")
)

# Workspace Header Row
hdr_col1, hdr_col2 = st.columns([3, 1])
with hdr_col1:
    st.markdown("<div class='title-primary'>ConfigPilot</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle-muted'>Configuration Reliability & Risk Analysis</div>", unsafe_allow_html=True)
with hdr_col2:
    if st.button("← Return to Intro", type="secondary"):
        st.session_state.landing_passed = False
        st.rerun()

# Status Bar
st.markdown(
    f"""
    <div class='status-bar'>
        <div>
            <code>{st.session_state.active_workspace}</code> &nbsp;·&nbsp;
            Dataset: <code>{st.session_state.dataset_name}</code>
        </div>
        <div>
            <span class='{badge_class}'>● {current_compat}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Helper: Render nested sidebar navigation row
def render_nav_row(page_label: str):
    is_active = (st.session_state.current_page == page_label)
    btn_kind = "primary" if is_active else "secondary"
    if st.sidebar.button(page_label, key=f"nav_btn_{page_label}", use_container_width=True, type=btn_kind):
        st.session_state.current_page = page_label
        st.rerun()

# Structural Nested Sidebar Navigation
st.sidebar.caption("SandDisk Challenge Copilot Prototype")

# Group 1: Dashboard
st.sidebar.markdown("<div class='nav-group-header'>Dashboard</div>", unsafe_allow_html=True)
render_nav_row("Dashboard Overview")
render_nav_row("Data Ingestion & Validation")

st.sidebar.markdown("<div class='nav-group-space'></div>", unsafe_allow_html=True)

# Group 2: Copilot
st.sidebar.markdown("<div class='nav-group-header'>Copilot</div>", unsafe_allow_html=True)
render_nav_row("Ask ConfigPilot")
render_nav_row("Failure Risk Prediction")
render_nav_row("Configuration Recommendations")

st.sidebar.markdown("<div class='nav-group-space'></div>", unsafe_allow_html=True)

# Group 3: Scenarios
st.sidebar.markdown("<div class='nav-group-header'>Scenarios</div>", unsafe_allow_html=True)
render_nav_row("What-If Analysis")
render_nav_row("Performance & Trade-offs")

st.sidebar.markdown("<div class='nav-group-space'></div>", unsafe_allow_html=True)

# Group 4: Reliability
st.sidebar.markdown("<div class='nav-group-header'>Reliability</div>", unsafe_allow_html=True)
render_nav_row("Failure Detective")
render_nav_row("What Changed?")
render_nav_row("Configuration Intelligence")
render_nav_row("Randomization & Determinism")

st.sidebar.markdown("<div class='nav-group-space'></div>", unsafe_allow_html=True)

# Group 5: Governance
st.sidebar.markdown("<div class='nav-group-header'>Governance</div>", unsafe_allow_html=True)
render_nav_row("Methodology & Audit")

st.sidebar.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
st.sidebar.caption("Random Forest (300 Trees) | Threshold: 29.00%")

navigation_mode = st.session_state.current_page


# ==============================================================================
# MODE: DASHBOARD OVERVIEW
# ==============================================================================
if navigation_mode == "Dashboard Overview":
    st.subheader("Dashboard Overview")
    st.markdown("System reliability metrics and active session configuration.")

    active_row = get_active_config_row(X_historical, st.session_state.active_config)
    active_pred = predict_failure_risk(model, best_threshold, active_row)

    st.markdown("##### Validated Model Benchmarks")
    st.caption("Validated on current synthetic challenge dataset (Untouched Test Set, N=988). Headline metrics remain stable across uploaded datasets.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Accuracy", "83.3%", "Validated Benchmark")
    m2.metric("Macro F1 Score", "0.706", "Val F1 Max Threshold")
    m3.metric("FAIL Recall", "66.92%", "87 of 130 Failures Detected")
    m4.metric("FAIL Precision", "41.63%", "TP=87, FP=122")

    st.markdown(
        """
        <div class='info-callout'>
            Precision Caveat: With 41.63% FAIL precision, roughly 3 in 5 flagged failures are false alarms (122 false positives out of 209 flagged runs).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Active Session Configuration")
    st.markdown(
        f"""
        <div class='panel-box'>
            <b>Active Parameters</b>: Queue Depth = <code>{st.session_state.active_config['queue_depth']}</code> |
            Thread Count = <code>{st.session_state.active_config['thread_count']}</code> |
            Cache = <code>{st.session_state.active_config['cache_size_mb']} MB</code> |
            Memory = <code>{st.session_state.active_config['memory_allocation_mb']} MB</code> |
            Workload = <code>{st.session_state.active_config['workload_type']}</code><br/>
            <span style='font-size:0.84rem; color:#9ca3af;'>Active configuration is stored in <code>st.session_state</code> for your current browser session.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_a, c_b, c_c = st.columns(3)
    with c_a:
        st.markdown("##### Copilot")
        st.write("• **Failure Risk Prediction**: Pre-failure risk score across 13 parameters.")
        st.write("• **Configuration Recommendations**: Top historical candidates under risk constraints.")
        st.write("• **Performance & Trade-offs**: Pareto curve of throughput vs risk.")
    with c_b:
        st.markdown("##### Scenarios")
        st.write("• **What-If Analysis**: Baseline vs scenario risk shift estimator.")
        st.write("• **Failure Detective**: Telemetry percentile ranks & sensitivity.")
        st.write("• **What Changed?**: Side-by-side run diff semantics.")
    with c_c:
        st.markdown("##### Reliability & Governance")
        st.write("• **Data Ingestion**: Schema verification with explicit compatibility tiers.")
        st.write("• **Feature Leakage Audit**: Structural audit confirming 60 pre-failure features.")
        st.write("• **Ask ConfigPilot**: Deterministic query engine.")

    st.markdown(
        """
        <div class='info-callout'>
            Challenge Data Context: Per hackathon organizer guidance, this prototype uses synthetically generated execution-log data (synthetic_execution_logs_10000.csv: 4,939 records, 114 columns, 4,289 PASS / 650 FAIL) because actual execution datasets cannot be distributed under NDA.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MODE: DATA INGESTION & VALIDATION
# ==============================================================================
elif navigation_mode == "Data Ingestion & Validation":
    st.subheader("Data Ingestion & Validation")
    st.markdown("Upload execution log CSV files to validate schema compatibility against model requirements.")

    i1, i2 = st.columns([1, 1])

    with i1:
        st.markdown("##### 1. Data Source")
        st.write(f"Active Dataset: `{st.session_state.dataset_name}`")

        uploaded_file = st.file_uploader("Upload Execution Log CSV", type=["csv"])

        if uploaded_file is not None:
            new_df, report = load_execution_data(uploaded_file)
            st.session_state.custom_df = new_df
            st.session_state.validation_report = report
            st.session_state.dataset_name = f"Uploaded CSV ({uploaded_file.name})"
            st.success(f"Ingested {report['total_rows']:,} records from `{uploaded_file.name}`.")
        else:
            if st.button("Reset to Default Synthetic Dataset"):
                st.session_state.custom_df = None
                st.session_state.validation_report = validate_dataset_schema(df_default)
                st.session_state.dataset_name = "synthetic_execution_logs_10000.csv (4,939 Runs)"
                st.info("Reset to default synthetic challenge dataset.")

    with i2:
        st.markdown("##### 2. Validation Report")
        v_report = (
            st.session_state.validation_report
            if st.session_state.validation_report
            else validate_dataset_schema(df_historical)
        )

        st.markdown(f"**Compatibility**: `{v_report['compatibility_status']}`")
        st.write(v_report["compatibility_message"])

        st.markdown(
            """
            <div class='info-callout'>
                Compatibility Rules:<br/>
                • SUPPORTED: 0 required model features missing AND target column present.<br/>
                • PARTIALLY COMPATIBLE: All 13 model-required parameters present, but non-model/optional analytics fields missing.<br/>
                • INCOMPATIBLE: ≥1 required model feature missing (predictions disabled).
            </div>
            """,
            unsafe_allow_html=True,
        )

        checks_df = pd.DataFrame(v_report["check_details"])
        st.dataframe(checks_df, width='stretch', hide_index=True)


# ==============================================================================
# MODE: ASK CONFIGPILOT
# ==============================================================================
elif navigation_mode == "Ask ConfigPilot":
    st.subheader("Ask ConfigPilot")
    st.markdown("Query ConfigPilot's decision engine using natural language questions.")

    st.markdown("##### Suggested Queries")
    q_cols1 = st.columns(3)
    suggested_q = None
    if q_cols1[0].button("What settings influence failure most?"):
        suggested_q = "What configuration settings influence failure the most?"
    if q_cols1[1].button("Which configurations perform best?"):
        suggested_q = "Which configurations have the best performance?"
    if q_cols1[2].button("How does dataset ingestion work?"):
        suggested_q = "How does dataset ingestion work?"

    q_cols2 = st.columns(3)
    if q_cols2[0].button("Are failures deterministic or random?"):
        suggested_q = "Are failures deterministic or random?"
    if q_cols2[1].button("Can future failures be predicted?"):
        suggested_q = "Can future failures be predicted?"
    if q_cols2[2].button("Recommend a config below 5% risk"):
        suggested_q = "Recommend a configuration below 5% predicted failure risk."

    user_query = st.text_input("Question:", value=suggested_q if suggested_q else "")

    if user_query:
        st.markdown("---")
        res = process_sanddisk_query(user_query, model, best_threshold, df_historical, X_historical)
        st.markdown(f"**Intent**: `{res['intent']}`")
        st.markdown(res["answer"])
        if res["data_type"] == "dataframe" and res["data"] is not None:
            st.dataframe(res["data"], width='stretch', hide_index=True)


# ==============================================================================
# MODE: FAILURE RISK PREDICTION
# ==============================================================================
elif navigation_mode == "Failure Risk Prediction":
    st.subheader("Failure Risk Prediction")
    st.markdown("Specify inputs across 13 controllable parameters to predict failure risk before running.")

    cfg = st.session_state.active_config
    col1, col2, col3 = st.columns(3)

    with col1:
        wl_options = df_historical["workload_type"].unique().tolist() if "workload_type" in df_historical.columns else ["MIXED_IO"]
        cfg["workload_type"] = st.selectbox("1. workload_type", wl_options, index=wl_options.index(cfg["workload_type"]) if cfg["workload_type"] in wl_options else 0)

        ti_options = df_historical["traffic_intensity"].unique().tolist() if "traffic_intensity" in df_historical.columns else ["MEDIUM"]
        cfg["traffic_intensity"] = st.selectbox("2. traffic_intensity", ti_options, index=ti_options.index(cfg["traffic_intensity"]) if cfg["traffic_intensity"] in ti_options else 0)

        cs_opts = [64, 128, 256, 512, 1024, 2048]
        cfg["cache_size_mb"] = st.select_slider("3. cache_size_mb", options=cs_opts, value=cfg["cache_size_mb"] if cfg["cache_size_mb"] in cs_opts else 512)

        cp_opts = df_historical["cache_policy"].unique().tolist() if "cache_policy" in df_historical.columns else ["ADAPTIVE"]
        cfg["cache_policy"] = st.selectbox("4. cache_policy", cp_opts, index=cp_opts.index(cfg["cache_policy"]) if cfg["cache_policy"] in cp_opts else 0)

        mem_opts = [1024, 2048, 4096, 8192, 12288, 16384]
        cfg["memory_allocation_mb"] = st.select_slider("5. memory_allocation_mb", options=mem_opts, value=cfg["memory_allocation_mb"] if cfg["memory_allocation_mb"] in mem_opts else 4096)

    with col2:
        mp_opts = df_historical["memory_policy"].unique().tolist() if "memory_policy" in df_historical.columns else ["DYNAMIC"]
        cfg["memory_policy"] = st.selectbox("6. memory_policy", mp_opts, index=mp_opts.index(cfg["memory_policy"]) if cfg["memory_policy"] in mp_opts else 0)

        cfg["thread_count"] = st.slider("7. thread_count", 1, 64, cfg["thread_count"])

        core_opts = [2, 4, 8, 16, 32]
        cfg["cpu_cores"] = st.select_slider("8. cpu_cores", options=core_opts, value=cfg["cpu_cores"] if cfg["cpu_cores"] in core_opts else 8)

        qd_opts = [1, 4, 8, 16, 32, 64, 96, 128, 192, 256]
        cfg["queue_depth"] = st.select_slider("9. queue_depth", options=qd_opts, value=cfg["queue_depth"] if cfg["queue_depth"] in qd_opts else 32)

    with col3:
        cfg["io_parallelism"] = st.slider("10. io_parallelism", 1, 64, cfg["io_parallelism"])

        req_opts = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        cfg["request_size_kb"] = st.select_slider("11. request_size_kb", options=req_opts, value=cfg["request_size_kb"] if cfg["request_size_kb"] in req_opts else 64)

        sch_opts = df_historical["io_scheduler"].unique().tolist() if "io_scheduler" in df_historical.columns else ["DYNAMIC"]
        cfg["io_scheduler"] = st.selectbox("12. io_scheduler", sch_opts, index=sch_opts.index(cfg["io_scheduler"]) if cfg["io_scheduler"] in sch_opts else 0)

        cfg["read_write_ratio"] = st.slider("13. read_write_ratio", 0.0, 100.0, float(cfg["read_write_ratio"]))

    st.session_state.active_config = cfg
    st.markdown("---")

    active_row = get_active_config_row(X_historical, cfg)
    pred_res = predict_failure_risk(model, best_threshold, active_row)
    fail_prob_pct = pred_res["failure_probability"] * 100.0

    st.markdown("##### Risk Evaluation")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Predicted Risk", f"{fail_prob_pct:.2f}%", f"Threshold = {best_threshold*100:.2f}%")
    r2.metric("Prediction", pred_res["prediction"])
    r3.metric("Risk Level", pred_res["risk_level"])
    risk_label_text = "LOW RISK" if pred_res["prediction"] == "PASS" else "HIGH RISK"
    r4.metric("Assessment", risk_label_text)

    st.markdown("##### Feature Sensitivity")
    sens_df = calculate_local_sensitivity(model, active_row, df_historical)
    if not sens_df.empty:
        st.dataframe(sens_df.head(6), width='stretch', hide_index=True)


# ==============================================================================
# MODE: CONFIGURATION RECOMMENDATIONS
# ==============================================================================
elif navigation_mode == "Configuration Recommendations":
    st.subheader("Configuration Recommendations")
    st.markdown("Set a maximum risk constraint to find top candidate historical configurations ranked by observed throughput.")

    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        risk_budget = st.slider("Maximum Acceptable Failure Risk Constraint (%)", 0.5, 20.0, 5.0, 0.5)
    with col_r2:
        selected_wl = st.selectbox("Filter Workload Type:", ["All Workloads"] + list(df_historical["workload_type"].unique() if "workload_type" in df_historical.columns else []))
        wl_filter = None if selected_wl == "All Workloads" else selected_wl

    rec_df, top_candidate = recommend_safe_configurations(
        model=model,
        historical_df=df_historical,
        max_risk_pct=risk_budget,
        workload_filter=wl_filter,
        top_n=10
    )

    if top_candidate is not None:
        st.markdown(
            f"""
            <div class='panel-box'>
                <b>Top Historical Candidate</b> (Risk ≤ {risk_budget}%, not a deployment guarantee)
            </div>
            """,
            unsafe_allow_html=True,
        )

        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Predicted Failure Risk", f"{top_candidate['predicted_risk_pct']:.2f}%", f"Constraint ≤ {risk_budget}%")
        tc2.metric("Historical Pass/Fail", str(top_candidate.get('pass_fail', 'PASS')))
        tc3.metric("Observed Throughput", f"{top_candidate.get('throughput_gbps', 0.0):.2f} Gbps")
        tc4.metric("Observed Latency", f"{top_candidate.get('average_latency_ms', 0.0):.2f} ms")

        if st.button("Apply Configuration", type="primary"):
            for feat in CONFIG_FEATURES:
                if feat in top_candidate.index:
                    st.session_state.active_config[feat] = top_candidate[feat]
            st.success("Active session configuration updated.")

        top_row_x = X_historical.loc[[top_candidate.name]]
        _, evidence = find_similar_configurations(top_row_x.iloc[0], df_historical, top_n=5)
        st.info(f"Evidence Strength: Supported by {evidence['total_similar_runs']} similar historical runs with a {evidence['historical_pass_rate_pct']}% historical pass rate and {evidence['average_throughput_gbps']} Gbps average throughput.")

        st.markdown(f"##### Recommended Candidates (Predicted Risk ≤ {risk_budget}%)")
        disp_cols = [c for c in ["run_id", "workload_type", "queue_depth", "cache_size_mb", "thread_count", "throughput_gbps", "average_latency_ms", "predicted_risk_pct", "pass_fail"] if c in rec_df.columns]
        st.dataframe(rec_df[disp_cols], width='stretch', hide_index=True)
    else:
        st.warning("No candidate configurations meet the specified risk constraint.")


# ==============================================================================
# MODE: PERFORMANCE & TRADE-OFFS
# ==============================================================================
elif navigation_mode == "Performance & Trade-offs":
    st.subheader("Performance & Trade-offs")
    st.markdown("Analyze trade-offs between observed historical throughput, latency, and predicted failure risk.")

    risk_budget_p = st.slider("Risk Cutoff Threshold (%)", 1.0, 25.0, 10.0, 0.5)

    rec_df, _ = recommend_safe_configurations(
        model=model,
        historical_df=df_historical,
        max_risk_pct=risk_budget_p,
        top_n=20
    )

    if not rec_df.empty:
        st.markdown("##### Pareto Configurations")
        disp_cols = [c for c in ["run_id", "workload_type", "queue_depth", "cache_size_mb", "thread_count", "throughput_gbps", "average_latency_ms", "predicted_risk_pct", "pass_fail"] if c in rec_df.columns]
        st.dataframe(rec_df[disp_cols], width='stretch', hide_index=True)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.scatter(rec_df["predicted_risk_pct"], rec_df["throughput_gbps"], color="#2563eb", s=60, alpha=0.8, edgecolors="black")
        ax.set_xlabel("Predicted Failure Risk (%)")
        ax.set_ylabel("Observed Throughput (Gbps)")
        ax.set_title("Observed Throughput vs Predicted Risk")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.warning("No historical configurations found meeting the selected risk cutoff.")


# ==============================================================================
# MODE: WHAT-IF ANALYSIS
# ==============================================================================
elif navigation_mode == "What-If Analysis":
    st.subheader("What-If Analysis")
    st.markdown("Simulate parameter changes against your active session configuration to evaluate risk deltas.")

    st.markdown(
        """
        <div class='info-callout'>
            Disclaimer: What-If simulation estimates configuration risk based on learned Random Forest model weights. It is a configuration-risk estimator, NOT a physical system simulator.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_side_by_side, tab_single = st.tabs(["Side-by-Side Comparison", "Single Parameter Sweep"])

    with tab_side_by_side:
        base_row = get_active_config_row(X_historical, st.session_state.active_config)
        base_res = predict_failure_risk(model, best_threshold, base_row)
        base_risk_pct = base_res["failure_probability"] * 100.0

        c_sc1, c_sc2 = st.columns(2)
        with c_sc1:
            st.markdown("##### Baseline Active Configuration")
            st.write(f"• Queue Depth: `{st.session_state.active_config['queue_depth']}`")
            st.write(f"• Thread Count: `{st.session_state.active_config['thread_count']}`")
            st.write(f"• Cache Size: `{st.session_state.active_config['cache_size_mb']} MB`")
            st.write(f"• Memory Allocation: `{st.session_state.active_config['memory_allocation_mb']} MB`")
            st.metric("Baseline Risk", f"{base_risk_pct:.2f}%", base_res["prediction"])

        with c_sc2:
            st.markdown("##### Scenario Inputs")
            sim_qd = st.slider("Simulated Queue Depth:", 1, 256, int(st.session_state.active_config['queue_depth']))
            sim_threads = st.slider("Simulated Thread Count:", 1, 64, int(st.session_state.active_config['thread_count']))

            changes = {"queue_depth": sim_qd, "thread_count": sim_threads}
            multi_res = multi_parameter_what_if(model, best_threshold, base_row, changes)
            new_risk_pct = multi_res["new_failure_probability"] * 100.0
            risk_delta = multi_res["percentage_point_change"]

            st.metric("Scenario Risk", f"{new_risk_pct:.2f}%", delta=f"{risk_delta:+.2f} pp", delta_color="inverse")

        st.markdown(f"**Verdict**: Risk Shift = `{risk_delta:+.2f} percentage points` (`{base_res['prediction']} ➔ {multi_res['new_prediction']}`)")

    with tab_single:
        sweep_feature = st.selectbox("Parameter to Sweep:", ["queue_depth", "read_write_ratio", "thread_count", "memory_allocation_mb", "cache_size_mb"])
        base_row = get_active_config_row(X_historical, st.session_state.active_config)

        if sweep_feature == "queue_depth":
            sweep_vals = [1, 4, 8, 16, 32, 64, 96, 128, 192, 256]
        elif sweep_feature == "read_write_ratio":
            sweep_vals = [10.0, 25.0, 50.0, 75.0, 90.0]
        elif sweep_feature == "thread_count":
            sweep_vals = [1, 4, 8, 16, 32, 48, 64]
        elif sweep_feature == "memory_allocation_mb":
            sweep_vals = [1024, 2048, 4096, 8192, 12288, 16384]
        else:
            sweep_vals = [64, 128, 256, 512, 1024, 2048]

        sweep_df = single_parameter_what_if(model, base_row, sweep_feature, sweep_vals)
        sweep_df["failure_probability_pct"] = sweep_df["failure_probability"] * 100.0

        disp_sweep = sweep_df[["feature", "simulated_value", "failure_probability_pct", "risk_level"]].rename(
            columns={"failure_probability_pct": "Predicted Failure Risk (%)"}
        )
        st.dataframe(disp_sweep, width='stretch', hide_index=True)

        fig, ax = plt.subplots(figsize=(7, 2.8))
        ax.plot(sweep_df["simulated_value"].astype(str), sweep_df["failure_probability_pct"], marker="o", color="#2563eb", linewidth=2)
        ax.axhline(best_threshold * 100, color="#dc2626", linestyle="--", label=f"Decision Boundary ({best_threshold*100:.1f}%)")
        ax.set_xlabel(sweep_feature)
        ax.set_ylabel("Predicted Failure Risk (%)")
        ax.set_title(f"Risk Impact of {sweep_feature}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)


# ==============================================================================
# MODE: FAILURE DETECTIVE
# ==============================================================================
elif navigation_mode == "Failure Detective":
    st.subheader("Failure Detective")
    st.markdown("Inspect historical execution runs with distinct separation between Model Predictions, Historical Outcomes, and Telemetry Ranks.")

    f_col1, f_col2 = st.columns([1, 3])
    with f_col1:
        run_filter = st.radio("Filter Runs:", ["All Runs", "FAIL Runs Only", "PASS Runs Only"])
        if run_filter == "FAIL Runs Only" and "pass_fail" in df_historical.columns:
            selectable_ids = df_historical[df_historical["pass_fail"] == "FAIL"]["run_id"].tolist()
        elif run_filter == "PASS Runs Only" and "pass_fail" in df_historical.columns:
            selectable_ids = df_historical[df_historical["pass_fail"] == "PASS"]["run_id"].tolist()
        elif "run_id" in df_historical.columns:
            selectable_ids = df_historical["run_id"].tolist()
        else:
            selectable_ids = list(range(len(df_historical)))

        selected_run_id = st.selectbox("Select Run ID:", selectable_ids)

    selected_row_df = df_historical[df_historical["run_id"] == selected_run_id] if "run_id" in df_historical.columns else df_historical.iloc[[selected_run_id]]

    if not selected_row_df.empty:
        row_data = selected_row_df.iloc[0]
        input_x = X_historical.loc[[row_data.name]]
        pred_res = predict_failure_risk(model, best_threshold, input_x)

        with f_col2:
            st.markdown(f"##### Run #{selected_run_id} Analysis")

            st.markdown("1. Model Prediction (Pre-Failure Features)")
            p1, p2, p3 = st.columns(3)
            p1.metric("Predicted Failure Risk", f"{pred_res['failure_probability'] * 100:.2f}%")
            p2.metric("Model Prediction", pred_res["prediction"])
            p3.metric("Model Status", pred_res["risk_level"])

            st.markdown("2. Ground-Truth Historical Outcome")
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Ground Truth", str(row_data.get("pass_fail", "N/A")))
            o2.metric("Failure Type", str(row_data.get("failure_type", "NONE")))
            o3.metric("Observed Throughput", f"{row_data.get('throughput_gbps', 0.0):.2f} Gbps")
            o4.metric("Observed Latency", f"{row_data.get('average_latency_ms', 0.0):.2f} ms")

            st.markdown("3. Telemetry Ranks & Percentile Observations")
            pct_df = get_feature_percentiles(row_data, df_historical)
            if not pct_df.empty:
                st.dataframe(pct_df, width='stretch', hide_index=True)


# ==============================================================================
# MODE: WHAT CHANGED?
# ==============================================================================
elif navigation_mode == "What Changed?":
    st.subheader("What Changed?")
    st.markdown("Compare two execution runs side-by-side to understand parameter differences, telemetry shifts, and risk divergence.")

    run_list = df_historical["run_id"].tolist() if "run_id" in df_historical.columns else list(range(len(df_historical)))

    c1, c2 = st.columns(2)
    with c1:
        run_a_id = st.selectbox("Select Run A (Baseline):", run_list, index=0)
    with c2:
        run_b_id = st.selectbox("Select Run B (Comparison):", run_list, index=min(1, len(run_list)-1))

    row_a = df_historical[df_historical["run_id"] == run_a_id].iloc[0] if "run_id" in df_historical.columns else df_historical.iloc[run_a_id]
    row_b = df_historical[df_historical["run_id"] == run_b_id].iloc[0] if "run_id" in df_historical.columns else df_historical.iloc[run_b_id]

    x_a = X_historical.loc[[row_a.name]]
    x_b = X_historical.loc[[row_b.name]]

    res_a = predict_failure_risk(model, best_threshold, x_a)
    res_b = predict_failure_risk(model, best_threshold, x_b)

    st.markdown("##### Executive Summary")
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    with comp_col1:
        st.write(f"**Run A (#{run_a_id})**")
        st.write(f"Predicted Risk: `{res_a['failure_probability'] * 100:.2f}%`")
        st.write(f"Actual Outcome: `{row_a.get('pass_fail', 'N/A')}`")

    with comp_col2:
        st.write(f"**Run B (#{run_b_id})**")
        st.write(f"Predicted Risk: `{res_b['failure_probability'] * 100:.2f}%`")
        st.write(f"Actual Outcome: `{row_b.get('pass_fail', 'N/A')}`")

    with comp_col3:
        risk_diff = (res_b['failure_probability'] - res_a['failure_probability']) * 100.0
        safer_config = "Run A" if risk_diff > 0 else ("Run B" if risk_diff < 0 else "Equal Risk")
        st.write(f"**Lower Predicted Risk**: {safer_config}")
        st.metric("Risk Delta", f"{abs(risk_diff):.2f} pp")

    st.markdown("##### Parameter Differences")
    diff_data = []
    all_cols = list(set(row_a.index).union(set(row_b.index)))
    for col in sorted(all_cols):
        if col in row_a.index and col in row_b.index:
            val_a = row_a[col]
            val_b = row_b[col]
            if str(val_a) != str(val_b):
                diff_data.append({
                    "Parameter": col,
                    "Run A": val_a,
                    "Run B": val_b,
                    "Category": "Controllable Config" if col in CONFIG_FEATURES else "Telemetry / Log",
                })

    if diff_data:
        st.dataframe(pd.DataFrame(diff_data), width='stretch', hide_index=True)


# ==============================================================================
# MODE: CONFIGURATION INTELLIGENCE
# ==============================================================================
elif navigation_mode == "Configuration Intelligence":
    st.subheader("Configuration Intelligence")
    st.markdown("Inspect global Random Forest feature importances categorized by Controllable Config vs Telemetry parameters.")

    df_imp = get_global_feature_importances(model, list(X_historical.columns))
    st.markdown(
        """
        <div class='info-callout'>
            Labeling Note: This table displays Model Reliance / Predictive Importance. Feature importances reflect decision weights in the trained Random Forest, NOT physical causal influence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Top Predictive Features")
    st.dataframe(df_imp.head(15), width='stretch', hide_index=True)

    fig, ax = plt.subplots(figsize=(7, 3))
    top_10 = df_imp.head(10)
    ax.barh(top_10["feature"][::-1], top_10["importance"][::-1], color="#2563eb")
    ax.set_xlabel("Model Predictive Importance")
    ax.set_title("Top 10 Feature Importances")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


# ==============================================================================
# MODE: RANDOMIZATION & DETERMINISM
# ==============================================================================
elif navigation_mode == "Randomization & Determinism":
    st.subheader("Randomization & Determinism")
    st.markdown("Inspect noise & randomization parameters ranked by model reliance.")

    df_imp = get_global_feature_importances(model, list(X_historical.columns))
    rand_features = ["timing_jitter_ms", "burst_probability", "fault_injection_probability", "voltage_variation_pct", "ambient_temperature_c", "request_arrival_rate"]
    rand_df = df_imp[df_imp["feature"].isin(rand_features)]
    st.dataframe(rand_df, width='stretch', hide_index=True)

    st.markdown("---")
    st.markdown("##### Seed Group Failure Rate Analysis")
    if "seed_group" in df_historical.columns and "pass_fail" in df_historical.columns:
        seed_grp_fails = df_historical.groupby('seed_group')['pass_fail'].apply(lambda x: (x == 'FAIL').mean() * 100).round(2)
        st.write(pd.DataFrame({"seed_group": seed_grp_fails.index, "failure_rate_pct": seed_grp_fails.values}))

    st.markdown(
        """
        <div class='info-callout'>
            Honest PoC Disclaimer: All 4,939 historical configurations represent unique parameter vectors (0 exact repeated trials). Insufficient repeated trials to establish deterministic behavior reliably for identical configurations.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MODE: METHODOLOGY & AUDIT
# ==============================================================================
elif navigation_mode == "Methodology & Audit":
    st.subheader("Methodology & Audit")

    leak_audit = run_feature_leakage_audit(df_default)

    st.markdown(f"##### Automated Feature Leakage Audit")
    st.code(leak_audit['audit_summary_text'])
    st.write("Post-outcome symptom features and ground-truth identifiers are strictly excluded prior to feature matrix construction:")
    st.write("• Excluded Identifiers (7): `run_id`, `timestamp`, `random_seed`, `seed_group`, `pass_fail`, `failure_type`, `execution_log`")
    st.write("• Excluded Auxiliary Columns (40): `config_aux_01` to `config_aux_40`")
    st.write("• Excluded Outcome Symptoms (7): `error_count`, `timeout_count`, `recovery_events`, `data_integrity_errors`, `watchdog_events`, `reliability_score`, `performance_score`")
    st.write("*Note: Correlation does not prove zero leakage; automated structural verification confirms that excluded metadata and post-outcome symptoms are absent from predictive inputs.*")

    st.markdown("##### Validated Model Benchmarks (Untouched Test Set, N=988)")
    st.write("Validated on current synthetic challenge dataset (`synthetic_execution_logs_10000.csv`: 4,939 records, 114 columns, 4,289 PASS / 650 FAIL):")
    st.write("• Overall Accuracy: `83.3%`")
    st.write("• Macro F1 Score: `0.706`")
    st.write("• FAIL Recall: `66.92%` (Detected 87 of 130 test failures)")
    st.write("• FAIL Precision: `41.63%` (TP=87, FP=122; 87 of 209 flagged runs were actual failures)")

    st.markdown(
        """
        <div class='info-callout'>
            Precision Caveat: With 41.63% FAIL precision, roughly 3 in 5 flagged failures are false alarms (122 false positives out of 209 flagged runs).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Session & Scope Disclosure")
    st.write("Session state (`st.session_state`) is isolated strictly to your current local browser session. No multi-tenant backend server or persistent database is present in this prototype.")

    st.markdown("##### Challenge Data Disclosure")
    st.write("Per hackathon organizer guidance, this prototype uses synthetically generated execution-log data because actual multi-GB execution datasets cannot be distributed under NDA and infrastructure constraints.")
