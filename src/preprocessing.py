"""
Preprocessing module for SandDisk AI Copilot.
Handles data loading, data leakage prevention, column identification,
and scikit-learn ColumnTransformer pipeline creation.
"""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# Identifiers & Context columns to drop
DROP_COLS: List[str] = [
    "run_id",
    "timestamp",
    "random_seed",
    "seed_group",
    "pass_fail",
    "failure_type",
    "execution_log",
]

# Unexplained auxiliary columns (config_aux_01 to config_aux_40)
AUX_COLS: List[str] = [f"config_aux_{i:02d}" for i in range(1, 41)]

# Post-failure outcome / symptom columns to drop to prevent Data Leakage
POST_FAILURE_COLS: List[str] = [
    "error_count",
    "timeout_count",
    "recovery_events",
    "data_integrity_errors",
    "watchdog_events",
    "reliability_score",
    "performance_score",
]

# Controllable configuration parameters for Copilot
CONFIG_FEATURES: List[str] = [
    "workload_type",
    "traffic_intensity",
    "cache_size_mb",
    "cache_policy",
    "memory_allocation_mb",
    "memory_policy",
    "thread_count",
    "cpu_cores",
    "queue_depth",
    "io_parallelism",
    "request_size_kb",
    "io_scheduler",
    "read_write_ratio",
]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load execution log dataset from CSV."""
    df = pd.read_csv(csv_path)
    return df


def prepare_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Extract leakage-free feature matrix X and target y (1 for FAIL, 0 for PASS).
    Returns (X, y, numeric_features, categorical_features).
    """
    # Define binary target: 1 = FAIL, 0 = PASS
    y = (df["pass_fail"] == "FAIL").astype(int)

    # Exclude leakage & metadata columns
    exclude_cols = set(DROP_COLS + AUX_COLS + POST_FAILURE_COLS)
    
    # Filter features present in df
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()

    # Identify numeric and categorical columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    return X, y, numeric_features, categorical_features


def create_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Create scikit-learn ColumnTransformer that passes numeric features through
    and OneHotEncodes categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    return preprocessor
