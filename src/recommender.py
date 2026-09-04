"""
Recommender module for SandDisk AI Copilot.
Recommends safer high-performance configurations under a user-defined risk constraint:
1. Evaluates candidate configurations using the trained model.
2. Filters out candidates with predicted failure probability > max_acceptable_risk.
3. Ranks remaining candidates by observed historical throughput.
4. Returns top recommended configuration candidates.

IMPORTANT LABELING:
These are top candidates under the selected predicted-risk constraint, NOT deployment guarantees.
"""

from typing import Any, Dict, Optional, Tuple
import pandas as pd

try:
    from preprocessing import prepare_feature_matrix
except ModuleNotFoundError:
    from src.preprocessing import prepare_feature_matrix


def recommend_safe_configurations(
    model: Any,
    historical_df: pd.DataFrame,
    max_risk_pct: float = 5.0,
    workload_filter: Optional[str] = None,
    top_n: int = 5
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Finds top high-throughput configuration candidates with predicted risk <= max_risk_pct.
    Returns (top_candidates_df, top_candidate_series).
    """
    max_risk = max_risk_pct / 100.0

    df_eval = historical_df.copy()

    # Optional workload filter
    if workload_filter and "workload_type" in df_eval.columns:
        df_eval = df_eval[df_eval["workload_type"] == workload_filter].copy()

    if df_eval.empty:
        return pd.DataFrame(), None

    # Prepare feature matrix for evaluation
    X_eval, _, _, _ = prepare_feature_matrix(df_eval)

    # Predict probabilities for all candidate rows
    predicted_probs = model.predict_proba(X_eval)[:, 1]
    df_eval["predicted_failure_probability"] = predicted_probs
    df_eval["predicted_risk_pct"] = predicted_probs * 100.0

    # Filter candidates meeting failure risk constraint
    safe_candidates = df_eval[df_eval["predicted_failure_probability"] <= max_risk].copy()

    if safe_candidates.empty:
        # Fallback to lowest risk candidates if none satisfy strict cutoff
        safe_candidates = df_eval.sort_values(by="predicted_failure_probability", ascending=True).head(top_n * 2)

    # Rank by throughput (highest performance first)
    if "throughput_gbps" in safe_candidates.columns:
        ranked_candidates = safe_candidates.sort_values(
            by=["throughput_gbps", "predicted_failure_probability"],
            ascending=[False, True]
        )
    else:
        ranked_candidates = safe_candidates.sort_values(
            by="predicted_failure_probability",
            ascending=True
        )

    top_n_df = ranked_candidates.head(top_n)
    top_candidate = top_n_df.iloc[0] if not top_n_df.empty else None

    return top_n_df, top_candidate
