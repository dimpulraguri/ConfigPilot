"""
Data Ingestion & Schema Validation Engine for ConfigPilot.
Handles CSV ingestion, data quality checks, schema verification,
and dataset compatibility classification.
"""

from typing import Any, Dict, List, Tuple, Union
import pandas as pd
from preprocessing import CONFIG_FEATURES, DROP_COLS, AUX_COLS, POST_FAILURE_COLS

# Critical features expected by the trained model (60 pre-failure features)
# These are extracted dynamically if present, but we define key required categories
REQUIRED_CONFIG_PARAMS: List[str] = CONFIG_FEATURES

# Expected target column
TARGET_COL: str = "pass_fail"


def validate_dataset_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates an uploaded execution log dataset against the ConfigPilot model schema.

    Returns a detailed validation dictionary with checks:
    - required_columns_present (bool)
    - missing_config_features (list)
    - missing_model_features (list)
    - duplicate_run_ids (int)
    - null_values_summary (dict)
    - outcome_available (bool)
    - compatibility_status ("SUPPORTED", "PARTIALLY COMPATIBLE", "INCOMPATIBLE")
    - compatibility_message (str)
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # 1. Required Controllable Parameters Check
    missing_config = [col for col in REQUIRED_CONFIG_PARAMS if col not in df.columns]
    
    # 2. Check Run ID & Duplicates
    has_run_id = "run_id" in df.columns
    duplicate_run_ids = 0
    if has_run_id:
        duplicate_run_ids = int(df["run_id"].duplicated().sum())

    # 3. Check Target Outcome Column
    outcome_available = TARGET_COL in df.columns
    pass_count = 0
    fail_count = 0
    if outcome_available:
        pass_count = int((df[TARGET_COL] == "PASS").sum())
        fail_count = int((df[TARGET_COL] == "FAIL").sum())

    # 4. Check Null / Missing Values
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()

    # 5. Determine Model Schema Compatibility
    # Exact Rule:
    # - SUPPORTED: 0 required model features missing AND ground-truth/analytics fields present.
    # - PARTIALLY COMPATIBLE: All 13 model-required parameters present, but non-model/optional analytics fields (e.g., pass_fail) missing.
    # - INCOMPATIBLE: >=1 required model feature missing (predictions disabled, since model cannot run without required inputs).
    if len(missing_config) > 0:
        compatibility_status = "INCOMPATIBLE"
        compatibility_message = (
            f"Prediction unavailable for this dataset because {len(missing_config)} required model feature(s) are missing: "
            f"{', '.join(missing_config[:5])}{'...' if len(missing_config) > 5 else ''}."
        )
    else:
        if outcome_available:
            compatibility_status = "SUPPORTED"
            compatibility_message = "Dataset is fully compatible with ConfigPilot model & analytics."
        else:
            compatibility_status = "PARTIALLY COMPATIBLE"
            compatibility_message = "All 13 required model parameters present. Failure risk prediction active, but ground-truth pass_fail target column is missing."

    validation_report = {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_config_features": missing_config,
        "has_run_id": has_run_id,
        "duplicate_run_ids": duplicate_run_ids,
        "outcome_available": outcome_available,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "cols_with_nulls": cols_with_nulls,
        "compatibility_status": compatibility_status,
        "compatibility_message": compatibility_message,
        "check_details": [
            {
                "check": "Controllable Parameters",
                "passed": len(missing_config) == 0,
                "detail": f"{len(REQUIRED_CONFIG_PARAMS) - len(missing_config)} / {len(REQUIRED_CONFIG_PARAMS)} present",
            },
            {
                "check": "Run ID Uniqueness",
                "passed": duplicate_run_ids == 0,
                "detail": f"{duplicate_run_ids} duplicates found" if duplicate_run_ids > 0 else "All Run IDs unique",
            },
            {
                "check": "Outcome Field (pass_fail)",
                "passed": outcome_available,
                "detail": f"{pass_count} PASS, {fail_count} FAIL" if outcome_available else "Not present",
            },
            {
                "check": "Data Completeness",
                "passed": len(cols_with_nulls) == 0,
                "detail": f"{len(cols_with_nulls)} columns with nulls" if cols_with_nulls else "Zero null values",
            },
        ],
    }

    return validation_report


def load_execution_data(file_source: Union[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safely load CSV dataset from file path or Streamlit UploadedFile handle,
    and run schema validation.
    """
    try:
        if isinstance(file_source, str):
            df = pd.read_csv(file_source)
        else:
            df = pd.read_csv(file_source)

        report = validate_dataset_schema(df)
        return df, report
    except Exception as e:
        report = {
            "total_rows": 0,
            "total_cols": 0,
            "compatibility_status": "INCOMPATIBLE",
            "compatibility_message": f"Failed to parse CSV file: {str(e)}",
            "check_details": [],
        }
        return pd.DataFrame(), report
