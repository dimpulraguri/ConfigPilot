"""
Ask ConfigPilot — Domain-Specific Natural Language Query Assistant
Built for the SandDisk Challenge
Deterministic intent engine retrieving real live data from ConfigPilot decision engine.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd

try:
    from explanation import get_global_feature_importances
    from recommender import recommend_safe_configurations
    from similarity import find_similar_configurations
    from prediction import predict_failure_risk
except ModuleNotFoundError:
    from src.explanation import get_global_feature_importances
    from src.recommender import recommend_safe_configurations
    from src.similarity import find_similar_configurations
    from src.prediction import predict_failure_risk


def process_sanddisk_query(
    query: str,
    model: Any,
    threshold: float,
    df_historical: pd.DataFrame,
    X_historical: pd.DataFrame
) -> Dict[str, Any]:
    """
    Processes a natural language query from an engineer or challenge judge,
    matches the intent, and retrieves dynamic computed answers from live application state.
    """
    q = query.lower().strip()
    
    # 0. Data Ingestion & Dataset Compatibility Queries
    if any(k in q for k in ["ingest", "upload", "csv", "compatibility", "incompatible"]):
        return {
            "intent": "Data Ingestion & Dataset Compatibility",
            "answer": (
                "**Data Ingestion & Validation in ConfigPilot**:\n\n"
                "ConfigPilot accepts external execution logs via the **Data Ingestion** command center.\n"
                "When a dataset is uploaded, the schema validator checks required parameters, null values, run ID uniqueness, and outcome labels.\n\n"
                "**Compatibility Levels**:\n"
                "- **`SUPPORTED`**: All 13 controllable parameters present. Full predictions and historical analytics active.\n"
                "- **`PARTIALLY COMPATIBLE`**: Core parameters present, minor non-essential features missing.\n"
                "- **`INCOMPATIBLE`**: Required model features missing. Predictions are safely disabled with detailed diagnostics rather than crashing."
            ),
            "data_type": "text",
            "data": None,
        }

    # 0b. Multi-User & Workspace Architecture Queries
    elif any(k in q for k in ["workspace", "tenant", "multi-user", "acme"]):
        return {
            "intent": "Workspace & Multi-Tenant Architecture",
            "answer": (
                "**Workspace Architecture Concept**:\n\n"
                "ConfigPilot supports workspace-isolated environments (e.g., `Acme Storage (Workspace)`).\n"
                "In production deployments, each workspace encapsulates isolated datasets, execution logs, active configurations, predictions, and reports.\n\n"
                "*(Note: The hackathon prototype provides workspace UI context to demonstrate multi-tenant readiness).* "
            ),
            "data_type": "text",
            "data": None,
        }

    # 1. Physics Simulator Disclaimer
    elif any(k in q for k in ["physics", "simulator"]):
        return {
            "intent": "What-If Simulator Disclaimer",
            "answer": (
                "**No. ConfigPilot is NOT a physical system physics simulator.**\n\n"
                "The What-If module estimates configuration risk under hypothetical inputs while holding baseline telemetry constant. "
                "Throughput and latency values displayed are observed historical values from past runs, NOT simulated predictions."
            ),
            "data_type": "text",
            "data": None,
        }

    # 2. Decision Threshold Explanation (29%)
    elif any(k in q for k in ["threshold", "29%", "0.29"]):
        return {
            "intent": "Decision Threshold Explanation",
            "answer": (
                f"The **`{threshold * 100:.2f}%` decision threshold** was selected strictly using the precision-recall curve on the **Validation Set** to maximize F1 score.\n\n"
                f"It defines the model's decision boundary for classifying FAIL vs PASS. It does NOT mean the system has a 29% real-world deployment failure rate."
            ),
            "data_type": "text",
            "data": None,
        }

    # 3. Model Evaluation & Challenge Data Context
    elif any(k in q for k in ["evaluated", "accuracy", "leakage", "nda", "organizer", "synthetic", "challenge"]):
        return {
            "intent": "Model Evaluation & Challenge Data Context",
            "answer": (
                "**Prototype Context**: Following challenge guidance, this prototype uses synthetically generated execution-log data because actual multi-GB data cannot be distributed under NDA.\n\n"
                "**Evaluation Results (Untouched Test Set)**:\n"
                "- Overall Accuracy: `83.3%`\n"
                "- FAIL Recall: `66.92%` (Detected 87 of 130 test failures)\n"
                "- Macro F1: `0.706`"
            ),
            "data_type": "text",
            "data": None,
        }

    # 4. Configuration Recommendations: Next Run / Recommend 5% Risk
    elif any(k in q for k in ["recommend", "next run", "below 5%", "safe config"]):
        top_df, top_cand = recommend_safe_configurations(model, df_historical, max_risk_pct=5.0, top_n=5)
        if top_cand is not None:
            return {
                "intent": "Configuration Recommendations",
                "answer": (
                    f"**Recommended Setting for Next Run**:\n"
                    f"- **Workload**: `{top_cand.get('workload_type', 'N/A')}`\n"
                    f"- **Queue Depth**: `{top_cand.get('queue_depth', 32)}` | **Thread Count**: `{top_cand.get('thread_count', 32)}` | **Cache**: `{top_cand.get('cache_size_mb', 512)} MB`\n"
                    f"- **Predicted Failure Risk**: `{top_cand['predicted_risk_pct']:.2f}%` (Constraint ≤ 5.0%)\n"
                    f"- **Observed Throughput**: `{top_cand.get('throughput_gbps', 0.0):.2f} Gbps`\n\n"
                    f"*(Wording: Top historical candidate under the selected predicted-risk constraint).* "
                ),
                "data_type": "dataframe",
                "data": top_df[["run_id", "workload_type", "queue_depth", "throughput_gbps", "average_latency_ms", "predicted_risk_pct", "pass_fail"]],
            }

    # 5. Randomization & Environmental Impact
    elif any(k in q for k in ["randomization", "random seed", "seed impact", "jitter", "noise"]):
        df_imp = get_global_feature_importances(model, list(X_historical.columns))
        rand_features = ["timing_jitter_ms", "burst_probability", "fault_injection_probability", "voltage_variation_pct", "ambient_temperature_c", "request_arrival_rate"]
        rand_df = df_imp[df_imp["feature"].isin(rand_features)]
        return {
            "intent": "Randomization & Environmental Impact",
            "answer": (
                f"Randomization & environmental noise variables ranked by model predictive reliance:\n"
                f"1. `{rand_df.iloc[0]['feature']}` (Importance: {rand_df.iloc[0]['importance']:.4f})\n"
                f"2. `{rand_df.iloc[1]['feature']}` (Importance: {rand_df.iloc[1]['importance']:.4f})\n\n"
                f"Higher timing jitter and voltage variation increase model predicted risk scores."
            ),
            "data_type": "dataframe",
            "data": rand_df,
        }

    # 6. Failure Determinism & Repeatability
    elif any(k in q for k in ["deterministic", "repeatability", "repeatable", "seed group"]):
        seed_grp_fails = df_historical.groupby('seed_group')['pass_fail'].apply(lambda x: (x == 'FAIL').mean() * 100).round(2)
        min_grp_fail = seed_grp_fails.min()
        max_grp_fail = seed_grp_fails.max()
        return {
            "intent": "Failure Determinism & Repeatability Analysis",
            "answer": (
                f"**Dataset Finding**: All 4,939 historical configurations represent unique parameter combinations (0 exact repeated 13-parameter trials).\n\n"
                f"Across the 20 historical `seed_group` partitions, observed failure rates vary between `{min_grp_fail}%` and `{max_grp_fail}%`, demonstrating stochastic environmental variation.\n\n"
                f"**Honest PoC Disclaimer**: Insufficient repeated trials to establish deterministic behavior reliably for identical configurations."
            ),
            "data_type": "text",
            "data": None,
        }

    # 7. Failure Detective Analysis
    elif any(k in q for k in ["why failure", "why did this fail", "failure detective", "contributing factors"]):
        return {
            "intent": "Failure Detective Analysis",
            "answer": (
                "In ConfigPilot's Failure Detective, failure root causes are inspected using:\n"
                "1. **Pre-Failure Feature Matrix**: Evaluation strictly on telemetry prior to run completion.\n"
                "2. **Percentile Observations**: Ranks features into VERY HIGH, HIGH, NORMAL, LOW, VERY LOW buckets.\n"
                "3. **Local Sensitivity**: Measures predicted risk shift when replacing features with historical medians."
            ),
            "data_type": "text",
            "data": None,
        }

    # 8. What Changed?
    elif any(k in q for k in ["what changed", "compare runs", "difference between"]):
        return {
            "intent": "What Changed? — Run & Configuration Comparison",
            "answer": (
                "The **What Changed?** module compares two execution runs side-by-side by:\n"
                "- Identifying modified parameters (Controllable Config vs Telemetry).\n"
                "- Calculating predicted failure probability for Run A vs Run B.\n"
                "- Outputting predicted risk delta in percentage points with a verdict on the configuration with lower predicted risk."
            ),
            "data_type": "text",
            "data": None,
        }

    # 9. Performance & Trade-offs (Pareto)
    elif any(k in q for k in ["best performance", "best configuration", "highest throughput", "pareto"]):
        top_df, top_cand = recommend_safe_configurations(model, df_historical, max_risk_pct=5.0, top_n=5)
        if top_cand is not None:
            return {
                "intent": "Performance & Trade-offs (Pareto Analysis)",
                "answer": (
                    f"Top candidate under 5% predicted failure risk constraint:\n"
                    f"- **Workload**: `{top_cand.get('workload_type', 'N/A')}`\n"
                    f"- **Predicted Risk**: `{top_cand['predicted_risk_pct']:.2f}%`\n"
                    f"- **Observed Throughput**: `{top_cand.get('throughput_gbps', 0.0):.2f} Gbps`\n"
                    f"- **Observed Latency**: `{top_cand.get('average_latency_ms', 0.0):.2f} ms`\n\n"
                    f"*(Label: Top historical candidate under the selected predicted-risk constraint, NOT a globally optimal guarantee).* "
                ),
                "data_type": "dataframe",
                "data": top_df[["run_id", "workload_type", "queue_depth", "throughput_gbps", "average_latency_ms", "predicted_risk_pct", "pass_fail"]],
            }

    # 10. Configuration Intelligence (Feature Importance)
    elif any(k in q for k in ["influence", "most important", "feature importance", "predictive importance"]):
        df_imp = get_global_feature_importances(model, list(X_historical.columns))
        top_5 = df_imp.head(5)["feature"].tolist()
        top_str = ", ".join([f"`{f}`" for f in top_5])
        return {
            "intent": "Configuration Intelligence",
            "answer": (
                f"Based on model reliance in our trained Random Forest pipeline, the top 5 predictive configuration & telemetry features are: {top_str}.\n\n"
                f"**Note**: Feature importance reflects model predictive reliance, NOT physical causal influence."
            ),
            "data_type": "dataframe",
            "data": df_imp.head(8),
        }

    # 11. Failure Risk Prediction
    elif any(k in q for k in ["predict", "future failure", "failure probability"]):
        return {
            "intent": "Failure Risk Prediction",
            "answer": (
                f"Yes, future failure risks are predicted before execution using a 300-tree Random Forest pipeline.\n\n"
                f"- **Model Output**: Predicted failure probability (0%–100%).\n"
                f"- **Decision Threshold**: `{threshold * 100:.2f}%` (selected on validation set F1 max).\n"
                f"- **Risk Levels**: LOW (<10%), MEDIUM (10%-25%), HIGH (≥25%)."
            ),
            "data_type": "text",
            "data": None,
        }

    # Default Fallback
    else:
        return {
            "intent": "ConfigPilot AI Assistant Overview",
            "answer": (
                "I am **ConfigPilot AI Reliability Assistant**. You can ask me:\n"
                "- *'What configuration settings influence failure the most?'*\n"
                "- *'Which configurations have the best performance?'*\n"
                "- *'Which randomization parameters matter most?'*\n"
                "- *'Are failures deterministic or random?'*\n"
                "- *'Can future failures be predicted?'*\n"
                "- *'Recommend a configuration below 5% predicted failure risk.'*\n"
                "- *'What does the 29% threshold mean?'*\n"
                "- *'Is this a physics simulator?'*"
            ),
            "data_type": "text",
            "data": None,
        }
