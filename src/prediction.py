"""
Prediction module for SandDisk AI Copilot.
Handles failure risk probability calculation, binary classification based on optimal threshold,
and risk level classification (LOW, MEDIUM, HIGH).
"""

from typing import Any, Dict, Union
import pandas as pd


def get_risk_level(failure_probability: float) -> str:
    """Classifies risk level based on predicted failure probability."""
    if failure_probability < 0.10:
        return "LOW"
    elif failure_probability < 0.25:
        return "MEDIUM"
    else:
        return "HIGH"


def predict_failure_risk(
    model: Any,
    threshold: float,
    input_data: pd.DataFrame
) -> Dict[str, Union[float, str]]:
    """
    Predict failure risk for an input feature row.
    Returns dictionary with:
    - failure_probability (float)
    - prediction ('PASS' or 'FAIL')
    - risk_level ('LOW', 'MEDIUM', 'HIGH')
    """
    # Predict failure probability (class 1)
    probs = model.predict_proba(input_data)
    failure_probability = float(probs[0, 1])

    # Apply optimal threshold selected during validation
    prediction = "FAIL" if failure_probability >= threshold else "PASS"

    # Determine risk category
    risk_level = get_risk_level(failure_probability)

    return {
        "failure_probability": failure_probability,
        "prediction": prediction,
        "risk_level": risk_level,
        "threshold_used": threshold,
    }
