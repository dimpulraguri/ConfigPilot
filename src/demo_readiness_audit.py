"""
ConfigPilot Final Demo Readiness Audit Script
Executes automated checks across all 9 evaluation sections.
"""

import os
import sys
import pandas as pd
import joblib

src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing import CONFIG_FEATURES, DROP_COLS, AUX_COLS, POST_FAILURE_COLS, prepare_feature_matrix
from ingestion import validate_dataset_schema, load_execution_data
from prediction import predict_failure_risk
from explanation import get_global_feature_importances, get_feature_percentiles, calculate_local_sensitivity
from what_if import single_parameter_what_if, multi_parameter_what_if
from recommender import recommend_safe_configurations
from similarity import find_similar_configurations
from chatbot import process_sanddisk_query

def audit_all():
    results = {}
    bugs = []

    print("==================================================")
    print("STARTING CONFIGPILOT FINAL DEMO READINESS AUDIT")
    print("==================================================")

    # 1. APP FUNCTIONALITY & MODULE IMPORTS
    try:
        import streamlit as st_test
        st_test.session_state.landing_passed = True
        import app
        results["1. APP FUNCTIONALITY"] = "PASS"
    except Exception as e:
        results["1. APP FUNCTIONALITY"] = f"FAIL: {e}"
        bugs.append(f"App import error: {e}")

    # 2. LIVE DATA FLOW
    try:
        model_path = os.path.join(project_root, "models", "sanddisk_failure_model.pkl")
        threshold_path = os.path.join(project_root, "models", "sanddisk_threshold.pkl")
        data_path = os.path.join(project_root, "data", "synthetic_execution_logs_10000.csv")

        model = joblib.load(model_path)
        threshold = float(joblib.load(threshold_path))
        df_raw = pd.read_csv(data_path)
        X_matrix, y_target, num_feats, cat_feats = prepare_feature_matrix(df_raw)

        # Check live prediction
        sample_row = X_matrix.iloc[[0]]
        pred_res = predict_failure_risk(model, threshold, sample_row)

        # Check local sensitivity
        sens_df = calculate_local_sensitivity(model, sample_row, df_raw)

        # Check recommendations
        rec_df, top_cand = recommend_safe_configurations(model, df_raw, max_risk_pct=5.0)

        assert pred_res["failure_probability"] > 0, "Prediction prob should be non-zero"
        assert not rec_df.empty, "Recommendation DataFrame should not be empty"
        assert not sens_df.empty, "Sensitivity DataFrame should not be empty"
        
        results["2. LIVE DATA FLOW"] = "PASS"
    except Exception as e:
        results["2. LIVE DATA FLOW"] = f"FAIL: {e}"
        bugs.append(f"Live data flow error: {e}")

    # 3. ACTIVE CONFIGURATION
    try:
        active_config = {
            "workload_type": "MIXED_IO",
            "traffic_intensity": "MEDIUM",
            "cache_size_mb": 512,
            "cache_policy": "ADAPTIVE",
            "memory_allocation_mb": 4096,
            "memory_policy": "DYNAMIC",
            "thread_count": 32,
            "cpu_cores": 8,
            "queue_depth": 32,
            "io_parallelism": 32,
            "request_size_kb": 64,
            "io_scheduler": "DYNAMIC",
            "read_write_ratio": 50.0,
        }
        # Verify propagation to custom row
        test_row = X_matrix.iloc[[0]].copy()
        for k, v in active_config.items():
            test_row.at[test_row.index[0], k] = v

        pred_active = predict_failure_risk(model, threshold, test_row)
        assert pred_active["prediction"] in ["PASS", "FAIL"], "Active config prediction invalid"

        # Apply recommendation test: Top candidate in rec_df must satisfy max_risk_pct (5.0%)
        top_cand_x = X_matrix.loc[[top_cand.name]]
        pred_updated = predict_failure_risk(model, threshold, top_cand_x)
        assert pred_updated["failure_probability"] <= 0.05, f"Top candidate should be under 5% risk constraint, got {pred_updated['failure_probability']*100:.2f}%!"
        
        results["3. ACTIVE CONFIGURATION"] = "PASS"
    except Exception as e:
        results["3. ACTIVE CONFIGURATION"] = f"FAIL: {e}"
        bugs.append(f"Active configuration error: {e}")

    # 4. DATA INGESTION EDGE CASES
    try:
        # A. Supported dataset
        rep_sup = validate_dataset_schema(df_raw)
        assert rep_sup["compatibility_status"] == "SUPPORTED", "Default CSV must be SUPPORTED"

        # B. Missing non-model/optional analytics column (PARTIALLY COMPATIBLE)
        df_missing_target = df_raw.drop(columns=["pass_fail"])
        rep_missing_target = validate_dataset_schema(df_missing_target)
        assert rep_missing_target["compatibility_status"] == "PARTIALLY COMPATIBLE", "Missing pass_fail should be PARTIALLY COMPATIBLE"

        # C. Incompatible dataset (Missing >= 1 required model feature)
        df_incomp = df_raw.drop(columns=["queue_depth"])
        rep_incomp = validate_dataset_schema(df_incomp)
        assert rep_incomp["compatibility_status"] == "INCOMPATIBLE", "Missing >=1 required model feature must be INCOMPATIBLE"

        # D. Duplicate run IDs
        df_dupes = df_raw.copy()
        df_dupes["run_id"] = 1001
        rep_dupes = validate_dataset_schema(df_dupes)
        assert rep_dupes["duplicate_run_ids"] > 0, "Duplicate run IDs should be detected"

        # E. Null values
        df_nulls = df_raw.copy()
        df_nulls.loc[0, "queue_depth"] = None
        rep_nulls = validate_dataset_schema(df_nulls)
        assert len(rep_nulls["cols_with_nulls"]) > 0, "Null values should be detected"

        results["4. DATA INGESTION"] = "PASS"
    except Exception as e:
        results["4. DATA INGESTION"] = f"FAIL: {e}"
        bugs.append(f"Data ingestion error: {e}")

    # 5. ASK CONFIGPILOT INTENT TEST
    try:
        queries = [
            ("Can future failures be predicted?", "Failure Risk Prediction"),
            ("What configuration settings influence failure the most?", "Configuration Intelligence"),
            ("What changed between runs?", "What Changed? — Run & Configuration Comparison"),
            ("Are failures deterministic or random?", "Failure Determinism & Repeatability Analysis"),
            ("Recommend a configuration below 5% predicted failure risk.", "Configuration Recommendations"),
            ("How does dataset ingestion work?", "Data Ingestion & Dataset Compatibility"),
            ("What does the 29% threshold mean?", "Decision Threshold Explanation"),
            ("Is this a physics simulator?", "What-If Simulator Disclaimer"),
        ]
        for q_text, expected_intent in queries:
            res = process_sanddisk_query(q_text, model, threshold, df_raw, X_matrix)
            assert res["intent"] == expected_intent, f"Query '{q_text}' mapped to '{res['intent']}', expected '{expected_intent}'"

        results["5. ASK CONFIGPILOT"] = "PASS"
    except Exception as e:
        results["5. ASK CONFIGPILOT"] = f"FAIL: {e}"
        bugs.append(f"Ask ConfigPilot error: {e}")

    # 6. ML INTEGRITY
    try:
        assert threshold == 0.2900, f"Threshold must be exactly 0.2900, got {threshold}"
        assert len(CONFIG_FEATURES) == 13, f"Must have 13 controllable parameters, got {len(CONFIG_FEATURES)}"
        
        # Verify no leakage columns are present in X_matrix
        leakage_cols = set(DROP_COLS + AUX_COLS + POST_FAILURE_COLS)
        overlap = set(X_matrix.columns).intersection(leakage_cols)
        assert len(overlap) == 0, f"Data leakage detected! Overlapping columns: {overlap}"

        # Verify feature count before one-hot encoding
        assert X_matrix.shape[1] == 60, f"Expected 60 features before OneHotEncoding, got {X_matrix.shape[1]}"

        results["6. ML INTEGRITY"] = "PASS"
    except Exception as e:
        results["6. ML INTEGRITY"] = f"FAIL: {e}"
        bugs.append(f"ML integrity error: {e}")

    # 7. DEPLOYMENT & RELATIVE PATHS
    try:
        app_file = os.path.join(project_root, "app.py")
        with open(app_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for hardcoded local Windows paths
        assert "C:\\Users\\" not in content, "Hardcoded C:\\Users\\ path found in app.py!"
        assert "c:\\Users\\" not in content, "Hardcoded c:\\Users\\ path found in app.py!"

        # Check requirements.txt
        req_file = os.path.join(project_root, "requirements.txt")
        assert os.path.exists(req_file), "requirements.txt missing!"
        
        results["7. DEPLOYMENT"] = "PASS"
    except Exception as e:
        results["7. DEPLOYMENT"] = f"FAIL: {e}"
        bugs.append(f"Deployment audit error: {e}")

    # 8. DOCUMENTATION AUDIT
    try:
        readme_file = os.path.join(project_root, "README.md")
        with open(readme_file, "r", encoding="utf-8") as f:
            readme_text = f.read().lower()

        assert "synthetic" in readme_text, "README missing synthetic dataset disclosure"
        assert "nda" in readme_text or "organizer" in readme_text, "README missing NDA/organizer constraint disclosure"
        assert "ingestion" in readme_text, "README missing data ingestion section"
        assert "physics" in readme_text or "estimator" in readme_text, "README missing physics simulator disclaimer"
        assert "guarantee" in readme_text or "recommend" in readme_text, "README missing recommendation non-guarantee disclaimer"

        results["8. DOCUMENTATION"] = "PASS"
    except Exception as e:
        results["8. DOCUMENTATION"] = f"FAIL: {e}"
        bugs.append(f"Documentation audit error: {e}")

    # 9. DEMO POLISH AUDIT
    try:
        app_file = os.path.join(project_root, "app.py")
        with open(app_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        found_markers = []
        for idx, l in enumerate(lines):
            if "print(" in l and not l.strip().startswith("#"):
                found_markers.append(f"Line {idx+1}: {l.strip()}")

        assert len(found_markers) == 0, f"Uncommented print statements found: {found_markers}"
        
        results["9. DEMO POLISH"] = "PASS"
    except Exception as e:
        results["9. DEMO POLISH"] = f"FAIL: {e}"
        bugs.append(f"Demo polish audit error: {e}")

    print("\n--------------------------------------------------")
    print("AUDIT RESULTS SUMMARY:")
    for section, status in results.items():
        print(f"  {section}: {status}")
    print("--------------------------------------------------")
    if bugs:
        print(f"BUGS FOUND ({len(bugs)}):")
        for b in bugs:
            print(f"  - {b}")
    else:
        print("BUGS FOUND: NONE")
    print("==================================================")

if __name__ == "__main__":
    audit_all()
