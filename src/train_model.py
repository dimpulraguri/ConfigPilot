"""
Model training and threshold optimization script for SandDisk AI Copilot.
Recreates exact leakage-free pipeline, stratified split, validation threshold selection,
untouched test set evaluation, and saves model artifacts.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
try:
    from preprocessing import CONFIG_FEATURES, create_preprocessor, load_dataset, prepare_feature_matrix
except ModuleNotFoundError:
    from src.preprocessing import CONFIG_FEATURES, create_preprocessor, load_dataset, prepare_feature_matrix


def train_and_evaluate(data_path: str, models_dir: str):
    """Executes leakage-free model training and evaluation."""
    print("=" * 60)
    print("SandDisk AI Copilot — Model Rebuild & Evaluation")
    print("=" * 60)

    # 1. Load Data
    print(f"\n[1/6] Loading dataset from: {data_path}")
    df = load_dataset(data_path)
    print(f"Total rows loaded: {len(df)}")
    print(f"Target distribution:\n{df['pass_fail'].value_counts()}")

    # 2. Extract Feature Matrix & Target
    print("\n[2/6] Preparing leakage-free feature matrix...")
    X, y, numeric_features, categorical_features = prepare_feature_matrix(df)
    print(f"Feature matrix shape: {X.shape} ({len(numeric_features)} numeric, {len(categorical_features)} categorical features)")
    print(f"Target distribution: PASS (0) = {(y == 0).sum()}, FAIL (1) = {(y == 1).sum()}")

    # 3. Stratified Train / Validation / Test Split
    print("\n[3/6] Splitting data into Train / Validation / Test (Stratified)...")
    # Step A: Split 80% train_val / 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Step B: Split train_val into 80% train / 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val
    )

    print(f"Training set:   {len(X_train)} samples ({y_train.sum()} failures)")
    print(f"Validation set: {len(X_val)} samples ({y_val.sum()} failures)")
    print(f"Test set:       {len(X_test)} samples ({y_test.sum()} failures)")

    # 4. Build & Train Random Forest Pipeline
    print("\n[4/6] Training Random Forest pipeline...")
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf_model)
    ])

    pipeline.fit(X_train, y_train)
    print("Model training completed successfully.")

    # 5. Threshold Selection on Validation Set ONLY
    print("\n[5/6] Optimizing classification threshold on VALIDATION set...")
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)

    # Compute F1 scores for each threshold (avoid divide by zero)
    f1_scores = np.zeros_like(thresholds)
    denom = precisions[:-1] + recalls[:-1]
    mask = denom > 0
    f1_scores[mask] = 2 * (precisions[:-1][mask] * recalls[:-1][mask]) / denom[mask]

    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_val_f1 = f1_scores[best_idx]
    best_val_prec = precisions[:-1][best_idx]
    best_val_rec = recalls[:-1][best_idx]

    print(f"Optimal Threshold (Val F1 Max): {best_threshold:.4f} ({best_threshold * 100:.2f}%)")
    print(f"Validation Metrics @ Threshold {best_threshold:.4f}:")
    print(f"  Precision: {best_val_prec:.4f}")
    print(f"  Recall:    {best_val_rec:.4f}")
    print(f"  F1 Score:  {best_val_f1:.4f}")

    # 6. Final Evaluation on Untouched Test Set
    print("\n[6/6] Evaluating model on UNTOUCHED Test Set...")
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    cm = confusion_matrix(y_test, test_preds)
    print("\nTest Set Confusion Matrix:")
    print(cm)
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_preds, target_names=["PASS (0)", "FAIL (1)"], digits=3))

    tn, fp, fn, tp = cm.ravel()
    print(f"Detailed Test Metrics:")
    print(f"  Total Failures in Test Set: {tp + fn}")
    print(f"  Detected Failures (True Positives): {tp}")
    print(f"  Missed Failures (False Negatives): {fn}")
    print(f"  False Alarms (False Positives): {fp}")
    print(f"  FAIL Recall: {tp / (tp + fn):.4f}")

    # Save Artifacts
    os.makedirs(models_dir, exist_ok=True)
    model_file = os.path.join(models_dir, "sanddisk_failure_model.pkl")
    threshold_file = os.path.join(models_dir, "sanddisk_threshold.pkl")
    config_features_file = os.path.join(models_dir, "sanddisk_config_features.pkl")

    joblib.dump(pipeline, model_file)
    joblib.dump(best_threshold, threshold_file)
    joblib.dump(CONFIG_FEATURES, config_features_file)

    print(f"\nArtifacts saved successfully:")
    print(f"  Model: {model_file}")
    print(f"  Threshold: {threshold_file}")
    print(f"  Config Features: {config_features_file}")
    print("=" * 60)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "synthetic_execution_logs_10000.csv")
    models_dir = os.path.join(project_root, "models")
    
    train_and_evaluate(data_path, models_dir)
