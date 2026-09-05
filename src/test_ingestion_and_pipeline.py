"""
Verification test script for Data Ingestion, Leakage Audit, Chatbot, and Pipeline capabilities.
"""

import os
import sys
import pandas as pd

src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import joblib
from ingestion import validate_dataset_schema, load_execution_data
from leakage_audit import run_feature_leakage_audit
from chatbot import process_sanddisk_query
from prediction import predict_failure_risk
from preprocessing import CONFIG_FEATURES, prepare_feature_matrix

def run_tests():
    print("==================================================")
    print("RUNNING CONFIGPILOT SYSTEM VERIFICATION TESTS")
    print("==================================================")

    # 1. Test Ingestion Validation on Default Dataset (SUPPORTED)
    csv_path = os.path.join(project_root, "data", "synthetic_execution_logs_10000.csv")
    df_raw, report = load_execution_data(csv_path)
    
    print(f"\n[1] Ingestion Validation (Default Dataset):")
    print(f"    - Total Rows: {report['total_rows']}")
    print(f"    - Compatibility: {report['compatibility_status']}")
    assert report['compatibility_status'] == "SUPPORTED", "Default dataset must be SUPPORTED!"

    # 2. Expanded Ingestion Edge Cases Test Suite
    print(f"\n[2] Expanded Ingestion Edge Cases Test Suite:")

    # Case A: Empty file / 0 rows dataframe
    df_empty = pd.DataFrame(columns=CONFIG_FEATURES + ["pass_fail"])
    rep_empty = validate_dataset_schema(df_empty)
    print(f"    - Empty DataFrame (0 rows): Compatibility = {rep_empty['compatibility_status']}")
    assert rep_empty['compatibility_status'] == "SUPPORTED", "Empty dataframe with correct schema should be SUPPORTED!"

    # Case B: Wrong dtypes (e.g. numeric columns passed as strings)
    df_wrong_dtypes = df_raw.copy()
    df_wrong_dtypes["queue_depth"] = df_wrong_dtypes["queue_depth"].astype(str) + "_invalid"
    rep_dtypes = validate_dataset_schema(df_wrong_dtypes)
    print(f"    - Wrong Dtypes: Compatibility = {rep_dtypes['compatibility_status']}")
    assert rep_dtypes['compatibility_status'] == "SUPPORTED", "Schema presence determines compatibility tier"

    # Case C: Extra / unexpected columns
    df_extra = df_raw.copy()
    df_extra["extra_unseen_column_x"] = "random_value"
    rep_extra = validate_dataset_schema(df_extra)
    print(f"    - Extra Columns: Compatibility = {rep_extra['compatibility_status']}")
    assert rep_extra['compatibility_status'] == "SUPPORTED", "Extra columns should not break SUPPORTED tier!"

    # Case D: Duplicate run IDs
    df_dupes = df_raw.copy()
    df_dupes["run_id"] = 1001
    rep_dupes = validate_dataset_schema(df_dupes)
    print(f"    - Duplicate Run IDs: Duplicates = {rep_dupes['duplicate_run_ids']}")
    assert rep_dupes['duplicate_run_ids'] > 0, "Duplicates must be detected!"

    # Case E: Out-of-range values
    df_out_range = df_raw.copy()
    df_out_range.loc[0, "thread_count"] = 99999
    rep_out_range = validate_dataset_schema(df_out_range)
    print(f"    - Out-of-Range Values: Compatibility = {rep_out_range['compatibility_status']}")

    # Case F: PARTIALLY COMPATIBLE upload (missing pass_fail)
    df_part = df_raw.drop(columns=["pass_fail"])
    rep_part = validate_dataset_schema(df_part)
    print(f"    - Missing Target Column: Compatibility = {rep_part['compatibility_status']}")
    assert rep_part['compatibility_status'] == "PARTIALLY COMPATIBLE", "Missing pass_fail must be PARTIALLY COMPATIBLE!"

    # Case G: Fully INCOMPATIBLE upload (missing >= 1 required model feature)
    df_incomp = df_raw.drop(columns=["queue_depth"])
    rep_incomp = validate_dataset_schema(df_incomp)
    print(f"    - Missing Required Model Feature (queue_depth): Compatibility = {rep_incomp['compatibility_status']}")
    assert rep_incomp['compatibility_status'] == "INCOMPATIBLE", "Missing required feature must be INCOMPATIBLE!"

    # 3. Test Feature Leakage Audit Engine
    leak_report = run_feature_leakage_audit(df_raw)
    print(f"\n[3] Automated Feature Leakage Audit:")
    print(f"    - Status: {leak_report['status']}")
    print(f"    - Summary: {leak_report['audit_summary_text']}")
    assert leak_report['status'] == "PASS", "Leakage audit must PASS!"

    # 4. Test Chatbot Intent Recognition
    model_path = os.path.join(project_root, "models", "sanddisk_failure_model.pkl")
    threshold_path = os.path.join(project_root, "models", "sanddisk_threshold.pkl")
    model = joblib.load(model_path)
    threshold = float(joblib.load(threshold_path))
    X_matrix, _, _, _ = prepare_feature_matrix(df_raw)

    q_res = process_sanddisk_query("How does dataset ingestion work?", model, threshold, df_raw, X_matrix)
    print(f"\n[4] Chatbot Ingestion Query Test:")
    print(f"    - Intent: {q_res['intent']}")
    assert q_res['intent'] == "Data Ingestion & Dataset Compatibility", "Chatbot intent mismatch!"

    print("\n==================================================")
    print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
