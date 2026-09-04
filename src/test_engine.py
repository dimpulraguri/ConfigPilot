"""
Verification script for SandDisk AI Engine modules.
Ensures all Decision Engine modules can be imported and executed without errors.
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import (
    AUX_COLS,
    CONFIG_FEATURES,
    DROP_COLS,
    POST_FAILURE_COLS,
    create_preprocessor,
    load_dataset,
    prepare_feature_matrix,
)
from prediction import get_risk_level, predict_failure_risk
from explanation import (
    EXPLANATION_FEATURES,
    calculate_local_sensitivity,
    compute_percentile_category,
    get_feature_percentiles,
    get_global_feature_importances,
)
from what_if import multi_parameter_what_if, single_parameter_what_if
from similarity import calculate_config_distance, find_similar_configurations
from recommender import recommend_safe_configurations


def test_imports_and_constants():
    print("Testing constants and imports...")
    assert len(DROP_COLS) == 7, "DROP_COLS mismatch"
    assert len(AUX_COLS) == 40, "AUX_COLS mismatch"
    assert len(POST_FAILURE_COLS) == 7, "POST_FAILURE_COLS mismatch"
    assert len(CONFIG_FEATURES) == 13, "CONFIG_FEATURES mismatch"
    assert len(EXPLANATION_FEATURES) == 15, "EXPLANATION_FEATURES mismatch"
    print("[OK] All imports and constants verified.")


def test_risk_levels():
    print("Testing risk level categorization...")
    assert get_risk_level(0.05) == "LOW"
    assert get_risk_level(0.15) == "MEDIUM"
    assert get_risk_level(0.30) == "HIGH"
    print("[OK] Risk levels verified.")


if __name__ == "__main__":
    test_imports_and_constants()
    test_risk_levels()
    print("SandDisk Engine basic verification passed.")
