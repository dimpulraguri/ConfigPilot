"""
Similarity Search engine for SandDisk AI Copilot.
Finds historically similar system configurations using a weighted combination of:
- 70% normalized numeric distance
- 30% categorical mismatch rate

Presents results as historical evidence, not deployment guarantees.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    from preprocessing import CONFIG_FEATURES
except ModuleNotFoundError:
    from src.preprocessing import CONFIG_FEATURES


def calculate_config_distance(
    target_config: pd.Series,
    candidate_config: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    min_max_dict: Dict[str, Tuple[float, float]]
) -> float:
    """
    Computes weighted similarity distance between two configuration parameter vectors.
    Returns float distance (0 = identical, 1 = maximum dissimilarity).
    """
    # 1. Numeric Normalized Euclidean Distance (70% weight)
    num_dists = []
    for col in numeric_cols:
        if col in target_config.index and col in candidate_config.index:
            val1 = float(target_config[col])
            val2 = float(candidate_config[col])
            c_min, c_max = min_max_dict.get(col, (0.0, 1.0))
            rng = c_max - c_min if (c_max - c_min) > 1e-9 else 1.0
            norm_diff = (val1 - val2) / rng
            num_dists.append(norm_diff ** 2)

    numeric_dist = np.sqrt(np.mean(num_dists)) if num_dists else 0.0

    # 2. Categorical Mismatch Rate (30% weight)
    cat_mismatches = []
    for col in categorical_cols:
        if col in target_config.index and col in candidate_config.index:
            cat_mismatches.append(1.0 if str(target_config[col]) != str(candidate_config[col]) else 0.0)

    categorical_mismatch = np.mean(cat_mismatches) if cat_mismatches else 0.0

    # Total weighted distance
    total_distance = 0.70 * numeric_dist + 0.30 * categorical_mismatch
    return float(total_distance)


def find_similar_configurations(
    target_row: pd.Series,
    historical_df: pd.DataFrame,
    top_n: int = 5
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Finds the top N nearest historical configurations to target_row.
    Returns (top_n_df, historical_evidence_dict).
    """
    # Filter config features
    config_cols = [c for c in CONFIG_FEATURES if c in historical_df.columns]
    
    num_cols = [c for c in config_cols if pd.api.types.is_numeric_dtype(historical_df[c])]
    cat_cols = [c for c in config_cols if not pd.api.types.is_numeric_dtype(historical_df[c])]

    # Compute min and max for normalization
    min_max_dict = {}
    for col in num_cols:
        min_max_dict[col] = (float(historical_df[col].min()), float(historical_df[col].max()))

    distances = []
    for idx, row in historical_df.iterrows():
        dist = calculate_config_distance(target_row, row, num_cols, cat_cols, min_max_dict)
        distances.append(dist)

    df_copy = historical_df.copy()
    df_copy["similarity_distance"] = distances
    top_df = df_copy.sort_values(by="similarity_distance", ascending=True).head(top_n)

    # Calculate historical evidence summary
    total_runs = len(top_df)
    pass_count = int((top_df["pass_fail"] == "PASS").sum()) if "pass_fail" in top_df.columns else total_runs
    fail_count = total_runs - pass_count
    pass_rate = (pass_count / total_runs * 100) if total_runs > 0 else 0.0

    avg_throughput = float(top_df["throughput_gbps"].mean()) if "throughput_gbps" in top_df.columns else 0.0
    avg_latency = float(top_df["average_latency_ms"].mean()) if "average_latency_ms" in top_df.columns else 0.0
    closest_dist = float(top_df["similarity_distance"].iloc[0]) if total_runs > 0 else 0.0

    evidence = {
        "total_similar_runs": total_runs,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "historical_pass_rate_pct": round(pass_rate, 1),
        "average_throughput_gbps": round(avg_throughput, 2),
        "average_latency_ms": round(avg_latency, 2),
        "closest_distance": round(closest_dist, 4),
    }

    return top_df, evidence
