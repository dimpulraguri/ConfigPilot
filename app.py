"""
ConfigPilot — AI Configuration & Reliability Copilot
Built for the SandDisk Challenge
Production-Grade Reliability Engineering SaaS Application
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
from prediction import predict_failure_risk
from preprocessing import CONFIG_FEATURES, prepare_feature_matrix
from recommender import recommend_safe_configurations
from similarity import find_similar_configurations
from what_if import multi_parameter_what_if, single_parameter_what_if

# ==============================================================================
# 1. Page Configuration & Professional SaaS Styling
# ==============================================================================
st.set_page_config(
    page_title="ConfigPilot — AI Configuration & Reliability Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-family: 'Inter', system-ui, sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        margin-bottom: 0.1rem;
    }
    .tagline {
        font-size: 1.1rem;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 0.5px;
        margin-bottom: 0.2rem;
    }
    .workspace-banner {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 0.75rem 1.25rem;
        border-radius: 8px;
        color: #cbd5e1;
        font-size: 0.92rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.2rem;
    }
    .hero-banner {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-left: 5px solid #3b82f6;
        padding: 1.1rem 1.4rem;
        border-radius: 10px;
        color: #f8fafc;
        font-size: 1.02rem;
        font-weight: 500;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    }
    .disclaimer-box {
        background-color: #0f172a;
        border-left: 4px solid #f59e0b;
        padding: 0.85rem 1.1rem;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #cbd5e1;
        margin: 1rem 0;
        line-height: 1.5;
    }
    .badge-supported {
        background-color: #166534;
        color: #dcfce7;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-partial {
        background-color: #854d0e;
        color: #fef08a;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-incompatible {
        background-color: #991b1b;
        color: #fee2e2;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .nav-header {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 1px;
        margin-top: 1.1rem;
        margin-bottom: 0.3rem;
    }
    .active-config-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. Session State & Baseline Initialization
# ==============================================================================
if "active_workspace" not in st.session_state:
    st.session_state.active_workspace = "Acme Storage (Production Workspace)"

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "Synthetic Challenge Dataset (4,939 Executions)"

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
# 3. Artifact Loading & Safe Caching
# ==============================================================================
@st.cache_resource
def load_configpilot_artifacts():
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, "models", "sanddisk_failure_model.pkl")
    threshold_path = os.path.join(project_root, "models", "sanddisk_threshold.pkl")
    data_path = os.path.join(project_root, "data", "synthetic_execution_logs_10000.csv")

    if not os.path.exists(model_path) or not os.path.exists(threshold_path):
        st.error("Model artifacts not found! Please ensure models directory exists.")
        st.stop()

    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}!")
        st.stop()

    model = joblib.load(model_path)
    threshold = float(joblib.load(threshold_path))
    df_raw = pd.read_csv(data_path)
    X_matrix, _, _, _ = prepare_feature_matrix(df_raw)

    return model, threshold, df_raw, X_matrix


# Load persistent artifacts
try:
    model, best_threshold, df_default, X_default = load_configpilot_artifacts()
except Exception as e:
    st.error(f"Error loading ConfigPilot artifacts: {e}")
    st.stop()

# Determine currently active dataset
if st.session_state.custom_df is not None:
    df_historical = st.session_state.custom_df
    try:
        X_historical, _, _, _ = prepare_feature_matrix(df_historical)
    except Exception:
        X_historical = X_default
else:
    df_historical = df_default
    X_historical = X_default


# Helper: construct input row DataFrame from active_config
def get_active_config_row(baseline_df: pd.DataFrame, config_dict: dict) -> pd.DataFrame:
    row = baseline_df.iloc[[0]].copy()
    for k, v in config_dict.items():
        if k in row.columns:
            row.at[row.index[0], k] = v
    return row


# ==============================================================================
# 4. Header & Sidebar Navigation Structure (5 Categories)
# ==============================================================================
st.markdown("<div class='main-title'>ConfigPilot</div>", unsafe_allow_html=True)
st.markdown("<div class='tagline'>AI Configuration & Reliability Copilot &nbsp;|&nbsp; Predict. Explain. Simulate. Recommend.</div>", unsafe_allow_html=True)

# Workspace Banner Bar
current_compat = (
    st.session_state.validation_report["compatibility_status"]
    if st.session_state.validation_report
    else "SUPPORTED"
)
badge_class = (
    "badge-supported" if current_compat == "SUPPORTED"
    else ("badge-partial" if current_compat == "PARTIALLY COMPATIBLE" else "badge-incompatible")
)

st.markdown(
    f"""
    <div class='workspace-banner'>
        <div>
            🏢 <b>Workspace</b>: <code>{st.session_state.active_workspace}</code> &nbsp;|&nbsp;
            📊 <b>Execution Data</b>: <code>{st.session_state.dataset_name}</code>
        </div>
        <div>
            <span class='{badge_class}'>● {current_compat}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Grouped SaaS Navigation
st.sidebar.title("ConfigPilot Command")
st.sidebar.caption("SandDisk Challenge AI Copilot Engine")

st.sidebar.markdown("<div class='nav-header'>Command Center</div>", unsafe_allow_html=True)
nav_cc = ["🏠 Dashboard Overview", "📥 Data Ingestion & Validation"]

st.sidebar.markdown("<div class='nav-header'>Copilot Capabilities</div>", unsafe_allow_html=True)
nav_copilot = ["💬 Ask ConfigPilot", "🚀 Failure Risk Prediction", "🏆 Configuration Recommendations"]

st.sidebar.markdown("<div class='nav-header'>Scenarios & Trade-offs</div>", unsafe_allow_html=True)
nav_scenarios = ["🧪 What-If Analysis", "📈 Performance & Trade-offs"]

st.sidebar.markdown("<div class='nav-header'>Reliability Intelligence</div>", unsafe_allow_html=True)
nav_reliability = ["🔍 Failure Detective", "🔀 What Changed?", "📊 Configuration Intelligence", "🎲 Randomization & Determinism"]

st.sidebar.markdown("<div class='nav-header'>Governance & Audit</div>", unsafe_allow_html=True)
nav_gov = ["📋 Methodology & Audit"]

all_nav_options = nav_cc + nav_copilot + nav_scenarios + nav_reliability + nav_gov
navigation_mode = st.sidebar.radio("Navigate:", all_nav_options, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Baseline Model**: Random Forest (300 Trees)")
st.sidebar.markdown(f"**Decision Boundary**: `{best_threshold * 100:.2f}%` (Val F1 Max)")
st.sidebar.markdown(f"**Loaded Executions**: `{len(df_historical):,}` runs")
st.sidebar.caption("ConfigPilot Prototype | Built for SandDisk Challenge")


# ==============================================================================
# MODE: 🏠 DASHBOARD OVERVIEW & COMMAND CENTER
# ==============================================================================
if navigation_mode == "🏠 Dashboard Overview":
    st.header("🏠 Engineering Command Center")
    st.markdown("Operational overview of system reliability metrics, active workspace configuration status, and dataset health.")

    # Calculate active configuration risk
    active_row = get_active_config_row(X_historical, st.session_state.active_config)
    active_pred = predict_failure_risk(model, best_threshold, active_row)
    active_risk_pct = active_pred["failure_probability"] * 100.0

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Active Dataset Size", f"{len(df_historical):,} executions", f"Status: {current_compat}")
    d2.metric("PASS / FAIL Ratio", f"{(df_historical['pass_fail']=='PASS').sum() if 'pass_fail' in df_historical.columns else 'N/A'} / {(df_historical['pass_fail']=='FAIL').sum() if 'pass_fail' in df_historical.columns else 'N/A'}", "Historical Log Distribution")
    d3.metric("Active Config Risk", f"{active_risk_pct:.2f}%", f"Level: {active_pred['risk_level']}")
    d4.metric("Model Threshold", f"{best_threshold * 100:.2f}%", "Val Set F1 Optimization")

    st.markdown("### ⚡ Active Workspace Configuration Context")
    st.markdown(
        f"""
        <div class='active-config-card'>
            <b>Active Configuration Vector</b>: Queue Depth = <code>{st.session_state.active_config['queue_depth']}</code> |
            Thread Count = <code>{st.session_state.active_config['thread_count']}</code> |
            Cache = <code>{st.session_state.active_config['cache_size_mb']} MB</code> |
            Memory = <code>{st.session_state.active_config['memory_allocation_mb']} MB</code> |
            Workload = <code>{st.session_state.active_config['workload_type']}</code><br/>
            <span style='font-size:0.88rem; color:#94a3b8;'>This active configuration is shared persistently across Failure Risk Prediction, What-If Analysis, and Recommendations.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🌟 Copilot Decision Engine Capabilities")
    c_a, c_b, c_c = st.columns(3)

    with c_a:
        st.markdown("#### ⚡ Configuration Copilot")
        st.write("• **Failure Risk Prediction**: Pre-failure evaluation of 13 controllable parameters.")
        st.write("• **Configuration Recommendations**: Lower-predicted-risk candidates under risk budgets.")
        st.write("• **Performance & Trade-offs**: Pareto curve of throughput vs predicted risk.")

    with c_b:
        st.markdown("#### 🧪 Scenarios & Detective")
        st.write("• **What-If Analysis**: Interactive baseline vs scenario risk deltas.")
        st.write("• **Failure Detective**: Telemetry percentile ranks and local sensitivity.")
        st.write("• **What Changed?**: Side-by-side run diff semantics.")

    with c_c:
        st.markdown("#### 🛡️ Reliability & Governance")
        st.write("• **Data Ingestion**: External CSV validation and compatibility inspector.")
        st.write("• **Configuration Intelligence**: Feature importances & model reliance.")
        st.write("• **Ask ConfigPilot**: Natural language intent query engine.")

    st.markdown(
        """
        <div class='disclaimer-box'>
            📝 <b>Challenge Context & Synthetic Data Disclaimer</b>:<br/>
            <i>Following challenge guidance, this prototype uses synthetically generated execution-log data because actual execution data cannot be distributed under NDA. Production deployment would validate this exact pipeline on live hardware logs.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MODE: 📥 DATA INGESTION & SCHEMA VALIDATION
# ==============================================================================
elif navigation_mode == "📥 Data Ingestion & Validation":
    st.header("📥 Data Ingestion & Schema Validation")
    st.markdown("Upload external execution logs to validate schema compatibility against the ConfigPilot ML model.")

    i1, i2 = st.columns([1, 1])

    with i1:
        st.subheader("1. Execution Log Data Source")
        st.markdown("**Current Dataset**: " + st.session_state.dataset_name)

        uploaded_file = st.file_uploader("Upload Execution Log CSV", type=["csv"])

        if uploaded_file is not None:
            new_df, report = load_execution_data(uploaded_file)
            st.session_state.custom_df = new_df
            st.session_state.validation_report = report
            st.session_state.dataset_name = f"Uploaded CSV ({uploaded_file.name})"
            st.success(f"Successfully ingested {report['total_rows']:,} records from `{uploaded_file.name}`.")
        else:
            if st.button("Reset to Default Synthetic Dataset"):
                st.session_state.custom_df = None
                st.session_state.validation_report = validate_dataset_schema(df_default)
                st.session_state.dataset_name = "Synthetic Challenge Dataset (4,939 Executions)"
                st.info("Reset to default challenge dataset.")

    with i2:
        st.subheader("2. Schema & Quality Validation Report")
        v_report = (
            st.session_state.validation_report
            if st.session_state.validation_report
            else validate_dataset_schema(df_historical)
        )

        st.markdown(f"### Compatibility Status: `{v_report['compatibility_status']}`")
        st.write(v_report["compatibility_message"])

        st.markdown("#### Diagnostic Checks Summary")
        checks_df = pd.DataFrame(v_report["check_details"])
        st.dataframe(checks_df, use_container_width=True, hide_index=True)

        if v_report["missing_config_features"]:
            st.warning(f"⚠️ Missing Controllable Parameters: `{', '.join(v_report['missing_config_features'])}`")


# ==============================================================================
# MODE: 💬 ASK CONFIGPILOT
# ==============================================================================
elif navigation_mode == "💬 Ask ConfigPilot":
    st.header("💬 Ask ConfigPilot — AI Reliability Assistant")
    st.markdown("Query ConfigPilot's decision engine using natural language questions to retrieve live computed answers.")

    st.markdown("**Suggested Engineering & Judge Queries:**")
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

    user_query = st.text_input("Type your question for ConfigPilot Assistant:", value=suggested_q if suggested_q else "")

    if user_query:
        st.markdown("---")
        res = process_sanddisk_query(user_query, model, best_threshold, df_historical, X_historical)

        st.markdown(f"### 🎯 Intent: `{res['intent']}`")
        st.markdown(res["answer"])

        if res["data_type"] == "dataframe" and res["data"] is not None:
            st.dataframe(res["data"], use_container_width=True, hide_index=True)


# ==============================================================================
# MODE: 🚀 FAILURE RISK PREDICTION
# ==============================================================================
elif navigation_mode == "🚀 Failure Risk Prediction":
    st.header("🚀 Failure Risk Prediction")
    st.markdown("Specify or adjust the active system configuration across 13 controllable parameters to predict failure risk before running.")

    st.subheader("1. Controllable Parameter Inputs (Syncs to Active Workspace State)")

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

    # Evaluate Active Configuration
    active_row = get_active_config_row(X_historical, cfg)
    pred_res = predict_failure_risk(model, best_threshold, active_row)
    fail_prob_pct = pred_res["failure_probability"] * 100.0

    st.subheader("2. ConfigPilot Risk Evaluation")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Predicted Failure Risk", f"{fail_prob_pct:.2f}%", f"Threshold = {best_threshold*100:.2f}%")
    r2.metric("Model Prediction", pred_res["prediction"])
    r3.metric("Risk Level", pred_res["risk_level"])

    risk_label_text = "LOW PREDICTED RISK" if pred_res["prediction"] == "PASS" else "HIGH PREDICTED RISK"
    r4.metric("Model Assessment", risk_label_text)

    # Local Sensitivity Analysis
    st.markdown("### 🔍 Sensitivity & Contributing Factors")
    sens_df = calculate_local_sensitivity(model, active_row, df_historical)
    if not sens_df.empty:
        st.dataframe(sens_df.head(6), use_container_width=True, hide_index=True)


# ==============================================================================
# MODE: 🏆 CONFIGURATION RECOMMENDATIONS
# ==============================================================================
elif navigation_mode == "🏆 Configuration Recommendations":
    st.header("🏆 Configuration Recommendations")
    st.markdown("Set a maximum risk constraint. ConfigPilot evaluates candidate historical configurations and presents Pareto candidates ranked by observed throughput.")

    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        risk_budget = st.slider("Maximum Acceptable Predicted Failure Risk Constraint (%)", 0.5, 20.0, 5.0, 0.5)
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
            """
            <div class='hero-banner'>
                🌟 <b>Top historical candidate under the selected predicted-risk constraint</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Predicted Failure Risk", f"{top_candidate['predicted_risk_pct']:.2f}%", f"Constraint ≤ {risk_budget}%")
        tc2.metric("Historical Pass/Fail", str(top_candidate.get('pass_fail', 'PASS')))
        tc3.metric("Observed Throughput", f"{top_candidate.get('throughput_gbps', 0.0):.2f} Gbps")
        tc4.metric("Observed Latency", f"{top_candidate.get('average_latency_ms', 0.0):.2f} ms")

        # Action: Apply as Active Configuration
        if st.button("⚡ Apply Top Candidate as Active Workspace Configuration"):
            for feat in CONFIG_FEATURES:
                if feat in top_candidate.index:
                    st.session_state.active_config[feat] = top_candidate[feat]
            st.success("Active workspace configuration updated! You can now test it in Prediction or What-If.")

        # Recommendation Evidence Strength
        top_row_x = X_historical.loc[[top_candidate.name]]
        _, evidence = find_similar_configurations(top_row_x.iloc[0], df_historical, top_n=5)
        st.info(f"📊 **Recommendation Evidence Strength**: Supported by `{evidence['total_similar_runs']}` similar historical runs with a `{evidence['historical_pass_rate_pct']}%` historical pass rate and `{evidence['average_throughput_gbps']} Gbps` average observed throughput.")

        st.markdown(f"#### Top 10 Recommended Configurations (Predicted Risk ≤ {risk_budget}%)")
        disp_cols = [c for c in ["run_id", "workload_type", "queue_depth", "cache_size_mb", "thread_count", "throughput_gbps", "average_latency_ms", "predicted_risk_pct", "pass_fail"] if c in rec_df.columns]
        st.dataframe(rec_df[disp_cols], use_container_width=True, hide_index=True)
    else:
        st.warning("No candidate configurations meet the specified risk constraint.")


# ==============================================================================
# MODE: 📈 PERFORMANCE & TRADE-OFFS
# ==============================================================================
elif navigation_mode == "📈 Performance & Trade-offs":
    st.header("📈 Performance & Trade-offs (Pareto Analysis)")
    st.markdown("Analyze trade-offs between observed historical throughput, latency, and predicted failure risk.")

    risk_budget_p = st.slider("Risk Cutoff Threshold (%)", 1.0, 25.0, 10.0, 0.5)

    rec_df, _ = recommend_safe_configurations(
        model=model,
        historical_df=df_historical,
        max_risk_pct=risk_budget_p,
        top_n=20
    )

    if not rec_df.empty:
        st.subheader("Pareto-Efficient Historical Configurations")
        disp_cols = [c for c in ["run_id", "workload_type", "queue_depth", "cache_size_mb", "thread_count", "throughput_gbps", "average_latency_ms", "predicted_risk_pct", "pass_fail"] if c in rec_df.columns]
        st.dataframe(rec_df[disp_cols], use_container_width=True, hide_index=True)

        st.markdown("### 📈 Throughput vs Predicted Failure Risk Trade-off Chart")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.scatter(rec_df["predicted_risk_pct"], rec_df["throughput_gbps"], color="#3b82f6", s=70, alpha=0.8, edgecolors="black")
        ax.set_xlabel("Predicted Failure Risk (%)")
        ax.set_ylabel("Observed Throughput (Gbps)")
        ax.set_title("Pareto Trade-off: Observed Throughput vs Predicted Failure Risk")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.warning("No historical configurations found meeting the selected risk cutoff.")


# ==============================================================================
# MODE: 🧪 WHAT-IF ANALYSIS
# ==============================================================================
elif navigation_mode == "🧪 What-If Analysis":
    st.header("🧪 What-If Analysis & Scenario Simulator")
    st.markdown("Simulate parameter changes against your baseline active configuration to evaluate risk deltas.")

    st.markdown(
        """
        <div class='disclaimer-box'>
            💡 <b>Methodological Limit & Disclaimer</b>: What-If simulation estimates configuration risk under hypothetical inputs while holding baseline telemetry constant. It is a configuration-risk estimator, NOT a physical system physics simulator.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_side_by_side, tab_single = st.tabs(["Side-by-Side Scenario Comparison", "Single Parameter Sweep"])

    with tab_side_by_side:
        st.subheader("Baseline vs Hypothetical Scenario Risk Shift")

        base_row = get_active_config_row(X_historical, st.session_state.active_config)
        base_res = predict_failure_risk(model, best_threshold, base_row)
        base_risk_pct = base_res["failure_probability"] * 100.0

        c_sc1, c_sc2 = st.columns(2)

        with c_sc1:
            st.markdown("#### Baseline Active Configuration")
            st.write(f"• **Queue Depth**: `{st.session_state.active_config['queue_depth']}`")
            st.write(f"• **Thread Count**: `{st.session_state.active_config['thread_count']}`")
            st.write(f"• **Cache Size**: `{st.session_state.active_config['cache_size_mb']} MB`")
            st.write(f"• **Memory Allocation**: `{st.session_state.active_config['memory_allocation_mb']} MB`")
            st.metric("Baseline Predicted Risk", f"{base_risk_pct:.2f}%", base_res["prediction"])

        with c_sc2:
            st.markdown("#### Modified Scenario Parameters")
            sim_qd = st.slider("Simulated Queue Depth:", 1, 256, int(st.session_state.active_config['queue_depth']))
            sim_threads = st.slider("Simulated Thread Count:", 1, 64, int(st.session_state.active_config['thread_count']))

            changes = {"queue_depth": sim_qd, "thread_count": sim_threads}
            multi_res = multi_parameter_what_if(model, best_threshold, base_row, changes)
            new_risk_pct = multi_res["new_failure_probability"] * 100.0
            risk_delta = multi_res["percentage_point_change"]

            st.metric("Scenario Predicted Risk", f"{new_risk_pct:.2f}%", delta=f"{risk_delta:+.2f} pp", delta_color="inverse")

        st.markdown(f"### Verdict: Risk Shift = `{risk_delta:+.2f} percentage points` (`{base_res['prediction']} ➔ {multi_res['new_prediction']}`)")

    with tab_single:
        st.subheader("Single Parameter Sensitivity Sweep")
        sweep_feature = st.selectbox("Select Parameter to Sweep:", ["queue_depth", "read_write_ratio", "thread_count", "memory_allocation_mb", "cache_size_mb"])

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
        st.dataframe(disp_sweep, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(sweep_df["simulated_value"].astype(str), sweep_df["failure_probability_pct"], marker="o", color="#3b82f6", linewidth=2)
        ax.axhline(best_threshold * 100, color="#ef4444", linestyle="--", label=f"Decision Boundary ({best_threshold*100:.1f}%)")
        ax.set_xlabel(sweep_feature)
        ax.set_ylabel("Predicted Failure Risk (%)")
        ax.set_title(f"What-If Risk Impact of {sweep_feature}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)


# ==============================================================================
# MODE: 🔍 FAILURE DETECTIVE
# ==============================================================================
elif navigation_mode == "🔍 Failure Detective":
    st.header("🔍 Failure Detective")
    st.markdown("Inspect historical execution runs with distinct separation between Model Predictions, Historical Outcomes, and Telemetry Ranks.")

    f_col1, f_col2 = st.columns([1, 3])

    with f_col1:
        run_filter = st.radio("Filter Historical Runs:", ["All Runs", "FAIL Runs Only", "PASS Runs Only"])
        if run_filter == "FAIL Runs Only" and "pass_fail" in df_historical.columns:
            selectable_ids = df_historical[df_historical["pass_fail"] == "FAIL"]["run_id"].tolist()
        elif run_filter == "PASS Runs Only" and "pass_fail" in df_historical.columns:
            selectable_ids = df_historical[df_historical["pass_fail"] == "PASS"]["run_id"].tolist()
        elif "run_id" in df_historical.columns:
            selectable_ids = df_historical["run_id"].tolist()
        else:
            selectable_ids = list(range(len(df_historical)))

        selected_run_id = st.selectbox("Select Run ID to Inspect:", selectable_ids)

    selected_row_df = df_historical[df_historical["run_id"] == selected_run_id] if "run_id" in df_historical.columns else df_historical.iloc[[selected_run_id]]

    if not selected_row_df.empty:
        row_data = selected_row_df.iloc[0]
        input_x = X_historical.loc[[row_data.name]]

        pred_res = predict_failure_risk(model, best_threshold, input_x)

        with f_col2:
            st.markdown(f"### Run #{selected_run_id} Root-Cause Investigation")

            # Box 1: Model Prediction
            st.markdown("#### 🤖 1. AI Model Prediction (Pre-Failure Features)")
            p1, p2, p3 = st.columns(3)
            p1.metric("Predicted Failure Risk", f"{pred_res['failure_probability'] * 100:.2f}%")
            p2.metric("Model Prediction", pred_res["prediction"])

            p_badge = "LOW PREDICTED RISK" if pred_res["prediction"] == "PASS" else "HIGH PREDICTED RISK"
            p3.metric("Model Status", p_badge)

            st.markdown("---")

            # Box 2: Ground Truth
            st.markdown("#### 📋 2. Ground-Truth Historical Outcome")
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Actual Ground Truth", str(row_data.get("pass_fail", "N/A")))
            o2.metric("Failure Type", str(row_data.get("failure_type", "NONE")))
            o3.metric("Observed Throughput", f"{row_data.get('throughput_gbps', 0.0):.2f} Gbps")
            o4.metric("Observed Latency", f"{row_data.get('average_latency_ms', 0.0):.2f} ms")

            st.markdown("---")

            # Box 3: Ranks & Percentiles
            st.markdown("#### 📊 3. Telemetry Ranks & Percentile Observations")
            pct_df = get_feature_percentiles(row_data, df_historical)
            if not pct_df.empty:
                st.dataframe(pct_df, use_container_width=True, hide_index=True)


# ==============================================================================
# MODE: 🔀 WHAT CHANGED?
# ==============================================================================
elif navigation_mode == "🔀 What Changed?":
    st.header("🔀 What Changed Between Executions?")
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

    st.markdown("### ⚡ Executive Comparison Summary")
    comp_col1, comp_col2, comp_col3 = st.columns(3)

    with comp_col1:
        st.markdown(f"#### Run A (#{run_a_id})")
        st.write(f"**Predicted Failure Risk**: `{res_a['failure_probability'] * 100:.2f}%`")
        st.write(f"**Model Prediction**: `{res_a['prediction']}`")
        st.write(f"**Actual Outcome**: `{row_a.get('pass_fail', 'N/A')}`")

    with comp_col2:
        st.markdown(f"#### Run B (#{run_b_id})")
        st.write(f"**Predicted Failure Risk**: `{res_b['failure_probability'] * 100:.2f}%`")
        st.write(f"**Model Prediction**: `{res_b['prediction']}`")
        st.write(f"**Actual Outcome**: `{row_b.get('pass_fail', 'N/A')}`")

    with comp_col3:
        risk_diff = (res_b['failure_probability'] - res_a['failure_probability']) * 100.0
        safer_config = "Run A" if risk_diff > 0 else ("Run B" if risk_diff < 0 else "Equal Risk")
        st.markdown(f"#### ConfigPilot Comparison Verdict")
        st.success(f"**Configuration with Lower Predicted Risk**: {safer_config}")
        st.metric("Predicted Risk Delta", f"{abs(risk_diff):.2f} percentage points")

    st.markdown("### 📋 Parameter Differences")
    diff_data = []
    all_cols = list(set(row_a.index).union(set(row_b.index)))
    for col in sorted(all_cols):
        if col in row_a.index and col in row_b.index:
            val_a = row_a[col]
            val_b = row_b[col]
            if str(val_a) != str(val_b):
                diff_data.append({
                    "Parameter / Metric": col,
                    "Run A Value": val_a,
                    "Run B Value": val_b,
                    "Category": "Controllable Config" if col in CONFIG_FEATURES else "System Telemetry / Log",
                })

    if diff_data:
        st.dataframe(pd.DataFrame(diff_data), use_container_width=True, hide_index=True)


# ==============================================================================
# MODE: 📊 CONFIGURATION INTELLIGENCE
# ==============================================================================
elif navigation_mode == "📊 Configuration Intelligence":
    st.header("📊 Configuration Intelligence")
    st.markdown("Inspect global Random Forest feature importances categorized by Controllable Config vs Telemetry parameters.")

    df_imp = get_global_feature_importances(model, list(X_historical.columns))

    st.markdown(
        """
        <div class='disclaimer-box'>
            ⚠️ <b>Labeling Note</b>: This table displays <b>Model Reliance / Predictive Importance</b>. Feature importances reflect model decision weights in the trained Random Forest, NOT physical causal influence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Top Predictive Features")
    st.dataframe(df_imp.head(15), use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    top_10 = df_imp.head(10)
    ax.barh(top_10["feature"][::-1], top_10["importance"][::-1], color="#8b5cf6")
    ax.set_xlabel("Model Predictive Importance")
    ax.set_title("Top 10 Model Predictive Features")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


# ==============================================================================
# MODE: 🎲 RANDOMIZATION & DETERMINISM
# ==============================================================================
elif navigation_mode == "🎲 Randomization & Determinism":
    st.header("🎲 Randomization Impact & Failure Determinism")

    st.subheader("1. Randomization Parameter Impact")
    st.markdown("Inspect noise & randomization parameters ranked by model reliance.")

    df_imp = get_global_feature_importances(model, list(X_historical.columns))
    rand_features = ["timing_jitter_ms", "burst_probability", "fault_injection_probability", "voltage_variation_pct", "ambient_temperature_c", "request_arrival_rate"]
    rand_df = df_imp[df_imp["feature"].isin(rand_features)]
    st.dataframe(rand_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("2. Failure Determinism Analysis")
    if "seed_group" in df_historical.columns and "pass_fail" in df_historical.columns:
        seed_grp_fails = df_historical.groupby('seed_group')['pass_fail'].apply(lambda x: (x == 'FAIL').mean() * 100).round(2)
        st.write(pd.DataFrame({"seed_group": seed_grp_fails.index, "failure_rate_pct": seed_grp_fails.values}))

    st.markdown(
        """
        <div class='disclaimer-box'>
            ℹ️ <b>Honest PoC Disclaimer</b>: All 4,939 historical configurations represent unique parameter vectors (0 exact repeated trials). Insufficient repeated trials to establish deterministic behavior reliably for identical configurations.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MODE: 📋 METHODOLOGY & AUDIT
# ==============================================================================
elif navigation_mode == "📋 Methodology & Audit":
    st.header("📋 Methodology & Audit")

    st.markdown(
        """
        ### 🛡️ Data Leakage Prevention
        Post-outcome symptom features and ground-truth identifiers were strictly excluded prior to feature matrix construction:
        - **Excluded Identifiers (7)**: `run_id`, `timestamp`, `random_seed`, `seed_group`, `pass_fail`, `failure_type`, `execution_log`
        - **Excluded Auxiliary Columns (40)**: `config_aux_01` to `config_aux_40`
        - **Excluded Outcome Symptoms (7)**: `error_count`, `timeout_count`, `recovery_events`, `data_integrity_errors`, `watchdog_events`, `reliability_score`, `performance_score`

        ### ⚙️ Stratified Train / Validation / Test Audit
        - **Total Dataset Size**: 4,939 validated synthetic logs (4,289 PASS, 650 FAIL)
        - **Train / Val / Test Split**: 3,160 Train, 791 Validation, 988 Untouched Test
        - **Validation Threshold Selection**: Optimal threshold (`0.2900` / `29.00%`) chosen strictly on **Validation Set F1 Max**.
        - **Untouched Test Set Evaluation**:
          - **Overall Accuracy**: `83.3%`
          - **Macro F1**: `0.706`
          - **FAIL Recall**: `66.92%` (Detected 87 of 130 test failures)

        ### 📝 Hackathon Challenge Context
        <i>Following challenge guidance, this prototype uses synthetically generated execution-log data because actual execution data cannot be distributed under NDA. Production deployment would require validation on real execution data.</i>
        """
    )
