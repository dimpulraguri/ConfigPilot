"""
Verification test script for Data Ingestion, Chatbot, and Pipeline capabilities.
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
from chatbot import process_sanddisk_query
from prediction import predict_failure_risk
from preprocessing import prepare_feature_matrix

def run_tests():
    print("==================================================")
    print("RUNNING CONFIGPILOT SYSTEM VERIFICATION TESTS")
    print("==================================================")

    # 1. Test Ingestion Validation on Default Dataset
    csv_path = os.path.join(project_root, "data", "synthetic_execution_logs_10000.csv")
    df_raw, report = load_execution_data(csv_path)
    
    print(f"\n[1] Ingestion Validation Test:")
    print(f"    - Total Rows: {report['total_rows']}")
    print(f"    - Compatibility: {report['compatibility_status']}")
    print(f"    - Message: {report['compatibility_message']}")
    assert report['compatibility_status'] == "SUPPORTED", "Default dataset should be SUPPORTED!"

    # 2. Test Incompatible Dataset Handling
    bad_df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    bad_report = validate_dataset_schema(bad_df)
    print(f"\n[2] Incompatible Dataset Test:")
    print(f"    - Compatibility: {bad_report['compatibility_status']}")
    print(f"    - Message: {bad_report['compatibility_message']}")
    assert bad_report['compatibility_status'] == "INCOMPATIBLE", "Bad dataset should be INCOMPATIBLE!"

    # 3. Test Chatbot Intent Recognition
    model_path = os.path.join(project_root, "models", "sanddisk_failure_model.pkl")
    threshold_path = os.path.join(project_root, "models", "sanddisk_threshold.pkl")
    model = joblib.load(model_path)
    threshold = float(joblib.load(threshold_path))
    X_matrix, _, _, _ = prepare_feature_matrix(df_raw)

    q_res = process_sanddisk_query("How does dataset ingestion work?", model, threshold, df_raw, X_matrix)
    print(f"\n[3] Chatbot Ingestion Query Test:")
    print(f"    - Intent: {q_res['intent']}")
    assert q_res['intent'] == "Data Ingestion & Dataset Compatibility", "Chatbot intent mismatch!"

    print("\n==================================================")
    print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
