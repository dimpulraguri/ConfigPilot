"""
What-If Simulator module for SandDisk AI Copilot.
Supports single-parameter parameter sweeps and multi-parameter hypothetical configuration changes.

IMPORTANT METHODOLOGICAL LIMITATION:
The simulator evaluates how the machine learning model's failure probability prediction
changes when inputs are altered while holding other baseline parameters constant.
It is a prototype risk-estimator, not a full physical system physics simulator.
"""

from typing import Any, Dict, List, Union
import pandas as pd


def single_parameter_what_if(
    model: Any,
    base_row: pd.DataFrame,
    feature_name: str,
    test_values: List[Union[int, float, str]]
) -> pd.DataFrame:
    """
    Sweeps a single feature across specified test values while holding all other columns constant.
    Returns DataFrame with test_value, predicted failure_probability, and risk_level.
    """
    if feature_name not in base_row.columns:
        raise ValueError(f"Feature '{feature_name}' not found in input data columns.")

    results = []
    for val in test_values:
        sim_row = base_row.copy()
        sim_row.at[sim_row.index[0], feature_name] = val

        prob = float(model.predict_proba(sim_row)[0, 1])

        if prob < 0.10:
            risk = "LOW"
        elif prob < 0.25:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        results.append({
            "feature": feature_name,
            "simulated_value": val,
            "failure_probability": prob,
            "risk_level": risk,
        })

    return pd.DataFrame(results)


def multi_parameter_what_if(
    model: Any,
    threshold: float,
    base_row: pd.DataFrame,
    changes: Dict[str, Union[int, float, str]]
) -> Dict[str, Any]:
    """
    Applies multiple parameter changes simultaneously to a baseline row.
    Returns dictionary with original risk, new risk, risk delta, and status shift.
    """
    sim_row = base_row.copy()

    # Original risk calculation
    orig_prob = float(model.predict_proba(base_row)[0, 1])
    orig_pred = "FAIL" if orig_prob >= threshold else "PASS"

    # Apply changes
    applied_changes = {}
    for feat, new_val in changes.items():
        if feat in sim_row.columns:
            orig_val = sim_row.at[sim_row.index[0], feat]
            sim_row.at[sim_row.index[0], feat] = new_val
            applied_changes[feat] = {"original": orig_val, "new": new_val}

    # New risk calculation
    new_prob = float(model.predict_proba(sim_row)[0, 1])
    new_pred = "FAIL" if new_prob >= threshold else "PASS"

    risk_delta = new_prob - orig_prob
    percentage_point_change = risk_delta * 100

    return {
        "original_failure_probability": orig_prob,
        "original_prediction": orig_pred,
        "new_failure_probability": new_prob,
        "new_prediction": new_pred,
        "risk_delta": risk_delta,
        "percentage_point_change": percentage_point_change,
        "applied_changes": applied_changes,
        "simulated_row": sim_row,
    }
