"""
Explanation module for SandDisk AI Copilot.
Provides:
1. Global Feature Importance extraction from Random Forest model.
2. Percentile-based feature value observation (VERY HIGH, HIGH, NORMAL, LOW, VERY LOW).
3. Local Sensitivity Analysis (measuring prediction probability shift when replacing a feature with its median).

IMPORTANT METHODOLOGICAL NOTE:
Percentiles and sensitivity measure model reliance and numerical shifts, NOT causal physical influence.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


# Key features to include in explanation analysis
EXPLANATION_FEATURES: List[str] = [
    "average_latency_ms",
    "queue_utilization_pct",
    "queue_depth",
    "p95_latency_ms",
    "p99_latency_ms",
    "memory_utilization_pct",
    "throughput_gbps",
    "power_consumption_w",
    "timing_jitter_ms",
    "read_write_ratio",
    "fault_injection_probability",
    "ambient_temperature_c",
    "burst_probability",
    "voltage_variation_pct",
    "request_arrival_rate",
]


def get_global_feature_importances(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Extract global feature importances from trained Random Forest classifier."""
    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_

    # Get feature names after ColumnTransformer transformation
    preprocessor = model.named_steps["preprocessor"]
    
    # Extract feature names from preprocessor
    num_cols = preprocessor.transformers_[0][2]
    cat_transformer = preprocessor.transformers_[1][1]
    cat_cols = preprocessor.transformers_[1][2]

    if hasattr(cat_transformer, "get_feature_names_out"):
        encoded_cat_cols = cat_transformer.get_feature_names_out(cat_cols).tolist()
    else:
        encoded_cat_cols = cat_cols

    transformed_feature_names = num_cols + encoded_cat_cols

    if len(importances) == len(transformed_feature_names):
        df_imp = pd.DataFrame({
            "feature": transformed_feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)
    else:
        # Fallback if mapping lengths differ
        df_imp = pd.DataFrame({
            "feature": [f"feature_{i}" for i in range(len(importances))],
            "importance": importances
        }).sort_values(by="importance", ascending=False)

    return df_imp


def compute_percentile_category(value: float, series: pd.Series) -> str:
    """Classifies a feature value into percentile buckets relative to historical dataset."""
    if series.empty or pd.isna(value):
        return "NORMAL"

    pct = (series < value).mean() * 100

    if pct >= 90:
        return "VERY HIGH"
    elif pct >= 75:
        return "HIGH"
    elif pct <= 10:
        return "VERY LOW"
    elif pct <= 25:
        return "LOW"
    else:
        return "NORMAL"


def get_feature_percentiles(input_row: pd.Series, historical_df: pd.DataFrame) -> pd.DataFrame:
    """Computes percentile ranks and qualitative tags for key explanation features."""
    results = []
    for feature in EXPLANATION_FEATURES:
        if feature in input_row.index and feature in historical_df.columns:
            val = input_row[feature]
            if isinstance(val, (int, float, np.number)):
                category = compute_percentile_category(float(val), historical_df[feature].dropna())
                pct_rank = (historical_df[feature] < val).mean() * 100
                results.append({
                    "feature": feature,
                    "value": float(val),
                    "percentile": round(pct_rank, 1),
                    "observation": category,
                })

    return pd.DataFrame(results)


def calculate_local_sensitivity(
    model: Any,
    input_row: pd.DataFrame,
    historical_df: pd.DataFrame,
    features_to_test: List[str] = None
) -> pd.DataFrame:
    """
    Performs local sensitivity analysis by replacing one numerical feature at a time
    with its historical median and measuring the change in predicted failure probability.
    """
    if features_to_test is None:
        features_to_test = [f for f in EXPLANATION_FEATURES if f in input_row.columns]

    base_prob = float(model.predict_proba(input_row)[0, 1])
    sensitivity_results = []

    for feature in features_to_test:
        if feature in input_row.columns and feature in historical_df.columns:
            if pd.api.types.is_numeric_dtype(historical_df[feature]):
                median_val = float(historical_df[feature].median())
                modified_row = input_row.copy()
                original_val = modified_row.at[modified_row.index[0], feature]
                modified_row.at[modified_row.index[0], feature] = median_val

                new_prob = float(model.predict_proba(modified_row)[0, 1])
                delta_prob = new_prob - base_prob

                sensitivity_results.append({
                    "feature": feature,
                    "original_value": original_val,
                    "median_value": median_val,
                    "original_risk": base_prob,
                    "new_risk": new_prob,
                    "risk_delta": delta_prob,
                    "impact": "INCREASED_RISK" if delta_prob > 0.005 else ("REDUCED_RISK" if delta_prob < -0.005 else "NEUTRAL")
                })

    df_sens = pd.DataFrame(sensitivity_results)
    if not df_sens.empty:
        df_sens["abs_delta"] = df_sens["risk_delta"].abs()
        df_sens = df_sens.sort_values(by="abs_delta", ascending=False).drop(columns=["abs_delta"])

    return df_sens
