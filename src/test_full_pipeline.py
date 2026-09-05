"""
End-to-end decision engine test script for SandDisk AI Copilot.
Loads trained artifacts and historical data, and verifies prediction, explanation,
what-if simulation, similarity search, and recommendation engines.
"""

import os
import joblib
import pandas as pd

from preprocessing import load_dataset, prepare_feature_matrix
from prediction import predict_failure_risk
from explanation import calculate_local_sensitivity, get_feature_percentiles, get_global_feature_importances
from what_if import multi_parameter_what_if, single_parameter_what_if
from similarity import find_similar_configurations
from recommender import recommend_safe_configurations


def test_full_pipeline():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "sanddisk_failure_model.pkl")
    threshold_path = os.path.join(project_root, "models", "sanddisk_threshold.pkl")
    data_path = os.path.join(project_root, "data", "synthetic_execution_logs_10000.csv")

    print("[1/5] Loading artifacts and dataset...")
    model = joblib.load(model_path)
    threshold = float(joblib.load(threshold_path))
    df = load_dataset(data_path)
    X, y, _, _ = prepare_feature_matrix(df)
    print(f"[OK] Artifacts loaded. Threshold = {threshold:.4f}, Rows = {len(df)}")

    print("\n[1b/5] Running Structural Feature Leakage Audit...")
    from leakage_audit import run_feature_leakage_audit
    leak_res = run_feature_leakage_audit(df)
    print(f"  {leak_res['audit_summary_text']}")
    assert leak_res["status"] == "PASS", "Leakage audit failed!"
    test_row = X.iloc[[0]]
    pred_res = predict_failure_risk(model, threshold, test_row)
    print(f"  Prediction: {pred_res['prediction']}, Prob = {pred_res['failure_probability']*100:.2f}%, Risk = {pred_res['risk_level']}")

    pct_df = get_feature_percentiles(df.iloc[0], df)
    print(f"  Feature percentiles computed: {len(pct_df)} features")

    sens_df = calculate_local_sensitivity(model, test_row, df)
    print(f"  Sensitivity items computed: {len(sens_df)}")

    print("\n[3/5] Testing What-If Simulator Engine...")
    sweep_df = single_parameter_what_if(model, test_row, "queue_depth", [1, 4, 16, 64, 128])
    print(f"  Single parameter sweep shape: {sweep_df.shape}")

    multi_res = multi_parameter_what_if(model, threshold, test_row, {"queue_depth": 64, "memory_utilization_pct": 80.0})
    print(f"  Multi parameter delta: {multi_res['percentage_point_change']:+.2f} percentage points")

    print("\n[4/5] Testing Similarity Engine...")
    top_sim, evidence = find_similar_configurations(test_row.iloc[0], df, top_n=5)
    print(f"  Similarity evidence: {evidence['total_similar_runs']} runs, Pass Rate = {evidence['historical_pass_rate_pct']}%")

    print("\n[5/5] Testing Recommender Engine...")
    top_recs, top_candidate = recommend_safe_configurations(model, df, max_risk_pct=5.0, top_n=5)
    print(f"  Recommended candidates found: {len(top_recs)}, Top throughput = {top_candidate['throughput_gbps']:.2f} Gbps")

    print("\n[OK] ALL SANDDISK DECISION ENGINE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_full_pipeline()
