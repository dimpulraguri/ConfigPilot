"""
ConfigPilot — AI Configuration & Reliability Copilot
Built for the SandDisk Challenge
Interactive Hackathon Application
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

# Import backend decision engine modules from src
from explanation import (
    EXPLANATION_FEATURES,
    calculate_local_sensitivity,
    get_feature_percentiles,
    get_global_feature_importances,
)
from prediction import predict_failure_risk
from preprocessing import CONFIG_FEATURES, prepare_feature_matrix
from recommender import recommend_safe_configurations
from similarity import find_similar_configurations
from what_if import multi_parameter_what_if, single_parameter_what_if
from chatbot import process_sanddisk_query

# ==============================================================================
# 1. Page Configuration & Custom CSS Styling
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
        font-size: 2.8rem;
        margin-bottom: 0.1rem;
    }
    .tagline {
        font-size: 1.15rem;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 0.5px;
        margin-bottom: 0.2rem;
    }
    .challenge-tag {
        font-size: 0.95rem;
        font-weight: 500;
        color: #3b82f6;
        margin-bottom: 0.9rem;
    }
    .hero-banner {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-left: 5px solid #3b82f6;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    }
    .disclaimer-box {
        background-color: #0f172a;
        border-left: 4px solid #f59e0b;
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #cbd5e1;
        margin: 1.2rem 0;
        line-height: 1.5;
    }
    .badge-low {
        background-color: #166534;
        color: #dcfce7;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.92rem;
    }
    .badge-high {
        background-color: #991b1b;
        color: #fee2e2;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. Artifact Loading & Safe Resource Caching
# ==============================================================================
@st.cache_resource
def load_configpilot_artifacts():
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, "models", "sanddisk_failure_model.pkl")
    threshold_path = os.path.join(project_root, "models", "sanddisk_threshold.pkl")
    data_path = os.path.join(project_root, "data", "synthetic_execution_logs_10000.csv")

    if not os.path.exists(model_path) or not os.path.exists(threshold_path):
        st.error("Model artifacts not found! Please run `python src/train_model.py` first.")
        st.stop()

    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}!")
        st.stop()

    model = joblib.load(model_path)
    threshold = float(joblib.load(threshold_path))
    df_raw = pd.read_csv(data_path)

    # Extract feature matrix X
    X_matrix, _, _, _ = prepare_feature_matrix(df_raw)

    return model, threshold, df_raw, X_matrix


# Load persistent artifacts
try:
    model, best_threshold, df_historical, X_historical = load_configpilot_artifacts()
except Exception as e:
    st.error(f"Error loading ConfigPilot artifacts: {e}")
    st.stop()


# ==============================================================================
# 3. Header Banner & Product Branding
# ==============================================================================
st.markdown("<div class='main-title'>ConfigPilot</div>", unsafe_allow_html=True)
st.markdown("<div class='tagline'>AI Configuration & Reliability Copilot &nbsp;|&nbsp; Predict. Explain. Simulate. Recommend.</div>", unsafe_allow_html=True)
st.markdown("<div class='challenge-tag'>💡 Built for the SandDisk Challenge</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='hero-banner'>
        🎯 <b>Core Product Principle</b>: <i>"Don't just monitor failures. Help engineers identify lower-predicted-risk configurations before the next run."</i>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Product Navigation
st.sidebar.title("ConfigPilot Navigation")
navigation_mode = st.sidebar.radio(
    "Select Capability:",
    [
        "🏠 Dashboard Overview",
        "💬 Ask ConfigPilot",
        "🚀 Failure Risk Prediction",
        "🏆 Configuration Recommendations",
        "📈 Performance & Trade-offs",
        "🧪 What-If Analysis",
        "🔍 Failure Detective",
        "🔀 What Changed?",
        "📊 Configuration Intelligence",
        "🎲 Randomization & Determinism",
        "📋 Methodology & Audit",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Baseline Model**: Random Forest (300 Trees)")
st.sidebar.markdown(f"**Decision Boundary**: `{best_threshold * 100:.2f}%` (Validation F1 Max)")
st.sidebar.markdown(f"**Dataset**: `{len(df_historical):,}` synthetic executions")
st.sidebar.caption("ConfigPilot Prototype | Built for the SandDisk Challenge")


# ==============================================================================
# MODE: 🏠 DASHBOARD OVERVIEW & EXECUTIVE SUMMARY
# ==============================================================================
if navigation_mode == "🏠 Dashboard Overview":
    st.header("🏠 Dashboard Overview")
    st.markdown("Executive summary of system reliability, model evaluation metrics, and dataset dimensions.")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Synthetic Execution Data", f"{len(df_historical):,} executions", "Synthetic PoC Dataset")
    s2.metric("PASS / FAIL Ratio", f"{(df_historical['pass_fail']=='PASS').sum()} / {(df_historical['pass_fail']=='FAIL').sum()}", "86.8% / 13.2%")
    s3.metric("Test Accuracy", "83.3%", f"FAIL Recall: 66.9%")
    s4.metric("Decision Boundary", f"{best_threshold * 100:.2f}%", "Val Set F1 Max")

    st.markdown("### 🌟 Capabilities Quick Navigation")
    c_a, c_b, c_c = st.columns(3)
    
    with c_a:
        st.markdown("#### ⚡ Configuration Copilot")
        st.write("• **Failure Risk Prediction**: Real-time evaluation of custom 13-parameter vectors.")
        st.write("• **Configuration Recommendations**: Lower-predicted-risk candidates under configurable risk constraints.")
        st.write("• **Performance & Trade-offs**: Pareto analysis of throughput vs risk.")
        st.write("• **What-If Analysis**: Single & multi-parameter sensitivity sweeps.")

    with c_b:
        st.markdown("#### 🛡️ Reliability Intelligence")
        st.write("• **Failure Detective**: Inspection of historical runs and root-cause factors.")
        st.write("• **What Changed?**: Side-by-side run comparison & parameter deltas.")
        st.write("• **Configuration Intelligence**: Predictive feature importance ranking.")
        st.write("• **Randomization & Determinism**: Seed group spread & noise impact.")

    with c_c:
        st.markdown("#### 💬 Assistant & Documentation")
        st.write("• **Ask ConfigPilot**: Domain-specific natural language query assistant.")
        st.write("• **Methodology & Audit**: Stratified split audit & data leakage rules.")

    st.markdown(
        """
        <div class='disclaimer-box'>
            📝 <b>Challenge Context & Synthetic Data Disclaimer</b>:<br/>
            <i>Following challenge guidance, this prototype uses synthetically generated execution-log data because the underlying real-world execution data cannot be distributed due to NDA and infrastructure constraints. The dataset is designed to demonstrate the intended analytics and ML workflow. Production deployment would require validation on real execution data.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MODE: 💬 ASK CONFIGPILOT (Natural Language Query Assistant)
# ==============================================================================
elif navigation_mode == "💬 Ask ConfigPilot":
    st.header("💬 Ask ConfigPilot — AI Reliability Assistant")
    st.markdown("Ask natural language questions to query ConfigPilot's decision engine and retrieve dynamic analytics from live application state.")

    st.markdown("**Suggested Engineering & Judge Queries:**")
    q_cols1 = st.columns(3)
    
    suggested_q = None
    if q_cols1[0].button("What settings influence failure most?"):
        suggested_q = "What configuration settings influence failure the most?"
    if q_cols1[1].button("Which configurations perform best?"):
        suggested_q = "Which configurations have the best performance?"
    if q_cols1[2].button("Which randomization parameters matter?"):
        suggested_q = "Which randomization parameters matter most?"

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
# MODE: FAILURE RISK PREDICTION
# ==============================================================================
elif navigation_mode == "🚀 Failure Risk Prediction":
    st.header("🚀 Failure Risk Prediction")
    st.markdown("Specify a system configuration across all 13 controllable parameters to predict failure risk before running the workload.")

    # Start with baseline feature row
    custom_input_row = X_historical.iloc[[0]].copy()

    st.subheader("1. System Configuration Inputs (13 Controllable Parameters)")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        wl_options = df_historical["workload_type"].unique().tolist() if "workload_type" in df_historical.columns else ["MIXED_IO"]
        val_wl = st.selectbox("1. workload_type", wl_options, index=0)
        custom_input_row.at[custom_input_row.index[0], "workload_type"] = val_wl

        ti_options = df_historical["traffic_intensity"].unique().tolist() if "traffic_intensity" in df_historical.columns else ["MEDIUM"]
        val_ti = st.selectbox("2. traffic_intensity", ti_options, index=min(1, len(ti_options)-1))
        custom_input_row.at[custom_input_row.index[0], "traffic_intensity"] = val_ti

        cs_opts = [64, 128, 256, 512, 1024, 2048]
        val_cs = st.select_slider("3. cache_size_mb", options=cs_opts, value=512)
        custom_input_row.at[custom_input_row.index[0], "cache_size_mb"] = val_cs

        cp_opts = df_historical["cache_policy"].unique().tolist() if "cache_policy" in df_historical.columns else ["ADAPTIVE"]
        val_cp = st.selectbox("4. cache_policy", cp_opts, index=0)
        custom_input_row.at[custom_input_row.index[0], "cache_policy"] = val_cp

        mem_opts = [1024, 2048, 4096, 8192, 12288, 16384]
        val_mem = st.select_slider("5. memory_allocation_mb", options=mem_opts, value=4096)
        custom_input_row.at[custom_input_row.index[0], "memory_allocation_mb"] = val_mem

    with col2:
        mp_opts = df_historical["memory_policy"].unique().tolist() if "memory_policy" in df_historical.columns else ["DYNAMIC"]
        val_mp = st.selectbox("6. memory_policy", mp_opts, index=0)
        custom_input_row.at[custom_input_row.index[0], "memory_policy"] = val_mp

        val_tc = st.slider("7. thread_count", 1, 64, 32)
        custom_input_row.at[custom_input_row.index[0], "thread_count"] = val_tc

        core_opts = [2, 4, 8, 16, 32]
        val_cores = st.select_slider("8. cpu_cores", options=core_opts, value=8)
        custom_input_row.at[custom_input_row.index[0], "cpu_cores"] = val_cores

        qd_opts = [1, 4, 8, 16, 32, 64, 96, 128, 192, 256]
        val_qd = st.select_slider("9. queue_depth", options=qd_opts, value=32)
        custom_input_row.at[custom_input_row.index[0], "queue_depth"] = val_qd

    with col3:
        val_iop = st.slider("10. io_parallelism", 1, 64, 32)
        custom_input_row.at[custom_input_row.index[0], "io_parallelism"] = val_iop

        req_opts = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        val_req = st.select_slider("11. request_size_kb", options=req_opts, value=64)
        custom_input_row.at[custom_input_row.index[0], "request_size_kb"] = val_req

        sch_opts = df_historical["io_scheduler"].unique().tolist() if "io_scheduler" in df_historical.columns else ["DYNAMIC"]
        val_sch = st.selectbox("12. io_scheduler", sch_opts, index=0)
        custom_input_row.at[custom_input_row.index[0], "io_scheduler"] = val_sch

        val_rw = st.slider("13. read_write_ratio", 0.0, 100.0, 50.0)
        custom_input_row.at[custom_input_row.index[0], "read_write_ratio"] = val_rw

    st.markdown("---")

    # Real-time prediction
    pred_res = predict_failure_risk(model, best_threshold, custom_input_row)
    fail_prob_pct = pred_res["failure_probability"] * 100.0

    st.subheader("2. ConfigPilot Risk Evaluation")
    
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Predicted Failure Risk", f"{fail_prob_pct:.2f}%", f"Threshold = {best_threshold*100:.2f}%")
    r2.metric("Model Prediction", pred_res["prediction"])
    r3.metric("Risk Level", pred_res["risk_level"])
    
    risk_label_text = "LOW PREDICTED RISK" if pred_res["prediction"] == "PASS" else "HIGH PREDICTED RISK"
    r4.metric("Model Assessment", risk_label_text)

    st.markdown(
        f"""
        <div class='disclaimer-box'>
            ℹ️ <b>Threshold Explanation</b>: The <b>{best_threshold * 100:.2f}% decision threshold</b> was selected strictly using the validation set precision-recall curve to maximize F1 score. It defines the PASS/FAIL classification boundary, NOT a 29% real-world deployment guarantee.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Local Sensitivity Analysis
    st.markdown("### 🔍 Sensitivity & Important Factors")
    sens_df = calculate_local_sensitivity(model, custom_input_row, df_historical)
    if not sens_df.empty:
        st.dataframe(sens_df.head(6), use_container_width=True, hide_index=True)


# ==============================================================================
# MODE: CONFIGURATION RECOMMENDATIONS
# ==============================================================================
elif navigation_mode == "🏆 Configuration Recommendations":
    st.header("🏆 Configuration Recommendations")
    st.markdown("Specify your maximum failure risk budget. ConfigPilot filters candidate configurations by predicted risk and ranks them by observed historical throughput.")

    risk_budget = st.slider("Maximum Acceptable Failure Risk Constraint (%)", 0.5, 20.0, 5.0, 0.5)
    selected_wl = st.selectbox("Filter Workload Type (Optional):", ["All Workloads"] + list(df_historical["workload_type"].unique()))
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

        # Recommendation Evidence Strength
        top_row_x = X_historical.loc[[top_candidate.name]]
        _, evidence = find_similar_configurations(top_row_x.iloc[0], df_historical, top_n=5)
        st.info(f"📊 **Recommendation Evidence Strength**: Supported by `{evidence['total_similar_runs']}` similar historical runs with a `{evidence['historical_pass_rate_pct']}%` historical pass rate and `{evidence['average_throughput_gbps']} Gbps` average observed throughput.")

        st.markdown("#### Top Candidate Parameter Settings")
        c_cols = [c for c in CONFIG_FEATURES if c in top_candidate.index]
        disp_top_cfg = pd.DataFrame([{"Parameter": c, "Value": top_candidate[c]} for c in c_cols])
        st.dataframe(disp_top_cfg, use_container_width=True, hide_index=True)

        st.markdown(f"#### Recommended Candidate Configurations (Predicted Risk ≤ {risk_budget}%)")
        disp_cols = [c for c in ["run_id", "workload_type", "queue_depth", "cache_size_mb", "thread_count", "throughput_gbps", "average_latency_ms", "predicted_risk_pct", "pass_fail"] if c in rec_df.columns]
        st.dataframe(rec_df[disp_cols], use_container_width=True, hide_index=True)
    else:
        st.warning("No candidate configurations meet the specified risk constraint.")


# ==============================================================================
# MODE: PERFORMANCE & TRADE-OFFS
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
# MODE: WHAT-IF ANALYSIS
# ==============================================================================
elif navigation_mode == "🧪 What-If Analysis":
    st.header("🧪 What-If Analysis")
    st.markdown("Modify single or multiple parameters to evaluate hypothetical risk shifts using the What-If decision engine.")

    st.markdown(
        """
        <div class='disclaimer-box'>
            💡 <b>Methodological Limit & Disclaimer</b>: What-If simulation estimates configuration risk under hypothetical inputs while holding baseline telemetry constant. It is a configuration-risk estimator, NOT a physical system physics simulator. Observed throughput and latency values reflect historical dataset records, NOT simulated predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_single, tab_multi = st.tabs(["Single Parameter Sweep", "Multi-Parameter Scenario"])

    with tab_single:
        st.subheader("Single Parameter Sensitivity Sweep")
        sweep_feature = st.selectbox("Select Parameter to Sweep:", ["queue_depth", "read_write_ratio", "thread_count", "memory_allocation_mb", "cache_size_mb"])
        
        baseline_row = X_historical.iloc[[0]].copy()

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

        sweep_df = single_parameter_what_if(model, baseline_row, sweep_feature, sweep_vals)
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

    with tab_multi:
        st.subheader("Multi-Parameter What-If Scenario")
        st.markdown("Example: Increase **queue_depth** to 64 and **memory_utilization_pct** to 80.0%")
        
        base_multi = X_historical.iloc[[0]].copy()
        
        m_qd = st.slider("Simulated queue_depth:", 1, 256, 64)
        m_mem_util = st.slider("Simulated memory_utilization_pct:", 10.0, 100.0, 80.0)

        changes_dict = {"queue_depth": m_qd, "memory_utilization_pct": m_mem_util}
        multi_res = multi_parameter_what_if(model, best_threshold, base_multi, changes_dict)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Original Risk", f"{multi_res['original_failure_probability']*100:.2f}%", multi_res['original_prediction'])
        mc2.metric("New Risk", f"{multi_res['new_failure_probability']*100:.2f}%", multi_res['new_prediction'])
        mc3.metric("Risk Delta", f"{multi_res['percentage_point_change']:+.2f} pp")
        mc4.metric("Status Shift", f"{multi_res['original_prediction']} ➔ {multi_res['new_prediction']}")


# ==============================================================================
# MODE: FAILURE DETECTIVE
# ==============================================================================
elif navigation_mode == "🔍 Failure Detective":
    st.header("🔍 Failure Detective")
    st.markdown("Inspect historical execution runs with clear separation between **Model Predictions**, **Historical Outcomes**, and **Sensitivity Analysis**.")

    f_col1, f_col2 = st.columns([1, 3])

    with f_col1:
        run_filter = st.radio("Filter Historical Runs:", ["All Runs", "FAIL Runs Only", "PASS Runs Only"])
        if run_filter == "FAIL Runs Only":
            selectable_ids = df_historical[df_historical["pass_fail"] == "FAIL"]["run_id"].tolist()
        elif run_filter == "PASS Runs Only":
            selectable_ids = df_historical[df_historical["pass_fail"] == "PASS"]["run_id"].tolist()
        else:
            selectable_ids = df_historical["run_id"].tolist()

        selected_run_id = st.selectbox("Select Run ID to Inspect:", selectable_ids)

    selected_row_df = df_historical[df_historical["run_id"] == selected_run_id]

    if not selected_row_df.empty:
        row_data = selected_row_df.iloc[0]
        input_x = X_historical.loc[[row_data.name]]

        pred_res = predict_failure_risk(model, best_threshold, input_x)

        with f_col2:
            st.markdown(f"### Run #{selected_run_id} Root-Cause Analysis")

            # Box 1: Model Prediction
            st.markdown("#### 🤖 1. AI Model Prediction (Pre-Failure Features)")
            p1, p2, p3 = st.columns(3)
            p1.metric("Predicted Failure Risk", f"{pred_res['failure_probability'] * 100:.2f}%")
            p2.metric("Model Prediction", pred_res["prediction"])
            
            p_badge = "LOW PREDICTED RISK" if pred_res["prediction"] == "PASS" else "HIGH PREDICTED RISK"
            p3.metric("Model Status", p_badge)

            st.markdown("---")

            # Box 2: Historical Outcome
            st.markdown("#### 📋 2. Ground-Truth Historical Outcome (Recorded Post-Run)")
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Actual Ground Truth", str(row_data.get("pass_fail", "N/A")))
            o2.metric("Failure Type", str(row_data.get("failure_type", "NONE")))
            o3.metric("Observed Throughput", f"{row_data.get('throughput_gbps', 0.0):.2f} Gbps")
            o4.metric("Observed Latency", f"{row_data.get('average_latency_ms', 0.0):.2f} ms")

            st.markdown("---")

            # Box 3: Important Contributing Factors & Percentile Observations
            st.markdown("#### 📊 3. Telemetry Ranks & Percentile Observations")
            pct_df = get_feature_percentiles(row_data, df_historical)
            if not pct_df.empty:
                st.dataframe(pct_df, use_container_width=True, hide_index=True)

            st.markdown(
                """
                <div class='disclaimer-box'>
                    ⚠️ <b>Non-Causal Disclaimer</b>: Feature percentiles and sensitivity indicate model reliance and statistical shifts in the synthetic dataset. They do NOT constitute physical causal proof.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Nearest matching historical configurations
            st.markdown("#### 👥 4. Similar Historical Runs")
            similar_df, evidence = find_similar_configurations(input_x.iloc[0], df_historical, top_n=5)
            st.write(f"**Historical Pass Rate**: `{evidence['historical_pass_rate_pct']}%` across `{evidence['total_similar_runs']}` nearest matching runs.")
            disp_cols = [c for c in ["run_id", "workload_type", "queue_depth", "throughput_gbps", "average_latency_ms", "pass_fail", "similarity_distance"] if c in similar_df.columns]
            st.dataframe(similar_df[disp_cols], use_container_width=True, hide_index=True)


# ==============================================================================
# MODE: WHAT CHANGED?
# ==============================================================================
elif navigation_mode == "🔀 What Changed?":
    st.header("🔀 What Changed Between Successful and Failed Executions?")
    st.markdown("Compare two execution runs side-by-side to understand parameter differences, telemetry shifts, and risk divergence.")

    c1, c2 = st.columns(2)
    with c1:
        run_a_id = st.selectbox("Select Run A (Baseline):", df_historical["run_id"].tolist(), index=0)
    with c2:
        run_b_id = st.selectbox("Select Run B (Comparison):", df_historical["run_id"].tolist(), index=min(1, len(df_historical)-1))

    row_a = df_historical[df_historical["run_id"] == run_a_id].iloc[0]
    row_b = df_historical[df_historical["run_id"] == run_b_id].iloc[0]

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
        st.write(f"**Actual Outcome**: `{row_a.get('pass_fail', 'N/A')}` (`{row_a.get('failure_type', 'NONE')}`)")
        st.write(f"**Observed Throughput**: `{row_a.get('throughput_gbps', 0.0):.2f} Gbps`")

    with comp_col2:
        st.markdown(f"#### Run B (#{run_b_id})")
        st.write(f"**Predicted Failure Risk**: `{res_b['failure_probability'] * 100:.2f}%`")
        st.write(f"**Model Prediction**: `{res_b['prediction']}`")
        st.write(f"**Actual Outcome**: `{row_b.get('pass_fail', 'N/A')}` (`{row_b.get('failure_type', 'NONE')}`)")
        st.write(f"**Observed Throughput**: `{row_b.get('throughput_gbps', 0.0):.2f} Gbps`")

    with comp_col3:
        risk_diff = (res_b['failure_probability'] - res_a['failure_probability']) * 100.0
        safer_config = "Run A" if risk_diff > 0 else ("Run B" if risk_diff < 0 else "Equal Risk")
        st.markdown(f"#### ConfigPilot Comparison Verdict")
        st.success(f"**Configuration with Lower Predicted Risk**: {safer_config}")
        st.metric("Predicted Risk Delta", f"{abs(risk_diff):.2f} percentage points", f"{'Run B higher risk' if risk_diff > 0 else 'Run A higher risk'}")

    st.markdown("### 📋 Parameter & Telemetry Differences")
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
    else:
        st.info("Runs A and B have identical parameter values.")

    st.markdown("### 📝 Execution Log Excerpts")
    log_a = row_a.get("execution_log", None)
    log_b = row_b.get("execution_log", None)
    
    if pd.notna(log_a) and str(log_a).strip():
        st.caption(f"Run #{run_a_id} Log: `{log_a}`")
    else:
        st.caption(f"Execution-log detail unavailable for Run #{run_a_id}.")


# ==============================================================================
# MODE: CONFIGURATION INTELLIGENCE
# ==============================================================================
elif navigation_mode == "📊 Configuration Intelligence":
    st.header("📊 Configuration Intelligence")
    st.markdown("Inspect global Random Forest feature importances categorized by Controllable Config vs Telemetry vs Randomization parameters.")

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
# MODE: RANDOMIZATION & DETERMINISM
# ==============================================================================
elif navigation_mode == "🎲 Randomization & Determinism":
    st.header("🎲 Randomization Impact & Failure Determinism")
    
    st.subheader("1. Randomization Parameter Impact")
    st.markdown("Inspect noise & randomization parameters (`timing_jitter_ms`, `burst_probability`, `fault_injection_probability`, `voltage_variation_pct`) ranked by model reliance.")
    
    df_imp = get_global_feature_importances(model, list(X_historical.columns))
    rand_features = ["timing_jitter_ms", "burst_probability", "fault_injection_probability", "voltage_variation_pct", "ambient_temperature_c", "request_arrival_rate"]
    rand_df = df_imp[df_imp["feature"].isin(rand_features)]
    st.dataframe(rand_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("2. Failure Determinism & Repeatability Analysis")
    seed_grp_fails = df_historical.groupby('seed_group')['pass_fail'].apply(lambda x: (x == 'FAIL').mean() * 100).round(2)
    min_fail = seed_grp_fails.min()
    max_fail = seed_grp_fails.max()

    st.markdown(f"**Dataset Observation**: Across 20 historical `seed_group` partitions (~250 runs per group), failure rates vary between `{min_fail}%` and `{max_fail}%` due to stochastic variation.")
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
# MODE: METHODOLOGY & AUDIT
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
        <i>Following challenge guidance, this prototype uses synthetically generated execution-log data because actual multi-GB data cannot be distributed under NDA. Production deployment would require validation on real execution data.</i>
        """
    )
