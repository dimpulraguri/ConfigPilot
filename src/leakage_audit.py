"""
Automated Feature Leakage Audit Engine for ConfigPilot.

Performs structural verification of input features:
1. Verifies exactly 60 pre-failure features enter the model matrix.
2. Structurally verifies that excluded metadata (IDs, timestamps, seeds, aux columns) cannot enter feature matrix.
3. Structurally verifies post-outcome/ground-truth symptom fields are absent from predictive inputs.
4. Confirms input schema and feature counts match frozen model requirements.
5. Computes diagnostic feature-target correlations for manual engineering review.

IMPORTANT: Diagnostic correlation flags indicate statistical association for review, NOT proof of zero leakage.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

from preprocessing import (
    CONFIG_FEATURES,
    DROP_COLS,
    AUX_COLS,
    POST_FAILURE_COLS,
    prepare_feature_matrix,
)

EXPECTED_PRE_FAILURE_FEATURE_COUNT = 60


def run_feature_leakage_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes a structural leakage audit on the dataset DataFrame and feature matrix.
    Returns audit status, checks breakdown, and correlation diagnostics.
    """
    # 1. Prepare Feature Matrix
    X, y, numeric_cols, categorical_cols = prepare_feature_matrix(df)
    feature_count = X.shape[1]

    # 2. Check 1: Feature Count Match
    count_passed = (feature_count == EXPECTED_PRE_FAILURE_FEATURE_COUNT)

    # 3. Check 2: Excluded Identifiers and Auxiliary Columns
    excluded_metadata = set(DROP_COLS + AUX_COLS)
    metadata_overlap = set(X.columns).intersection(excluded_metadata)
    metadata_passed = (len(metadata_overlap) == 0)

    # 4. Check 3: Post-Outcome Symptom Fields
    post_outcome_set = set(POST_FAILURE_COLS)
    post_outcome_overlap = set(X.columns).intersection(post_outcome_set)
    post_outcome_passed = (len(post_outcome_overlap) == 0)

    # 5. Check 4: Controllable Configuration Features Present
    missing_config = [c for c in CONFIG_FEATURES if c not in X.columns]
    config_passed = (len(missing_config) == 0)

    # 6. Diagnostic Correlation Analysis (Flag features with |r| > 0.5 for manual review)
    diagnostic_flags = []
    if "pass_fail" in df.columns:
        y_binary = (df["pass_fail"] == "FAIL").astype(int)
        for col in numeric_cols:
            if col in X.columns and X[col].nunique() > 1:
                corr = float(np.corrcoef(X[col].fillna(0), y_binary)[0, 1])
                if abs(corr) > 0.5:
                    diagnostic_flags.append({
                        "feature": col,
                        "correlation": round(corr, 4),
                        "note": "High feature-target correlation (|r| > 0.5) flagged for manual engineering review."
                    })

    overall_passed = (count_passed and metadata_passed and post_outcome_passed and config_passed)

    audit_summary_text = (
        "Leakage audit: PASS — 60 intended pre-failure features verified; excluded/post-outcome fields absent."
        if overall_passed else
        f"Leakage audit: FAIL — Issues detected (Count match: {count_passed}, Metadata excluded: {metadata_passed}, Symptoms excluded: {post_outcome_passed})."
    )

    return {
        "status": "PASS" if overall_passed else "FAIL",
        "audit_summary_text": audit_summary_text,
        "feature_count": feature_count,
        "expected_feature_count": EXPECTED_PRE_FAILURE_FEATURE_COUNT,
        "metadata_overlap": list(metadata_overlap),
        "post_outcome_overlap": list(post_outcome_overlap),
        "missing_config_features": missing_config,
        "diagnostic_flags": diagnostic_flags,
        "checks": [
            {
                "check": "Pre-Failure Feature Count",
                "passed": count_passed,
                "detail": f"{feature_count} / {EXPECTED_PRE_FAILURE_FEATURE_COUNT} features verified",
            },
            {
                "check": "Excluded Identifiers & Auxiliary Columns",
                "passed": metadata_passed,
                "detail": f"{len(metadata_overlap)} metadata columns present in feature matrix",
            },
            {
                "check": "Post-Outcome Symptom Fields Excluded",
                "passed": post_outcome_passed,
                "detail": f"{len(post_outcome_overlap)} symptom columns present in feature matrix",
            },
            {
                "check": "Controllable Configuration Schema",
                "passed": config_passed,
                "detail": f"{len(CONFIG_FEATURES) - len(missing_config)} / {len(CONFIG_FEATURES)} parameters present",
            },
        ],
    }


if __name__ == "__main__":
    import os
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    data_path = os.path.join(project_root, "data", "synthetic_execution_logs_10000.csv")
    df = pd.read_csv(data_path)
    res = run_feature_leakage_audit(df)
    print(res["audit_summary_text"])
    print("Checks:")
    for c in res["checks"]:
        print(f" - {c['check']}: {'PASS' if c['passed'] else 'FAIL'} ({c['detail']})")
    if res["diagnostic_flags"]:
        print("Diagnostic Flags:")
        for f in res["diagnostic_flags"]:
            print(f" - {f['feature']}: correlation = {f['correlation']}")
