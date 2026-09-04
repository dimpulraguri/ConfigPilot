# Implementation Plan: SandDisk AI Configuration & Reliability Copilot

Migrating the **SandDisk — AI Configuration & Reliability Copilot** hackathon project from Google Colab to a structured, reproducible local VS Code project environment.

## 1. Workspace Inspection Findings

We inspected the working directory (`c:\Users\dimpu\OneDrive\Documents\SandDisk`):
* **Dataset Present**: `synthetic_execution_logs_10000.csv` (4,939 rows × 114 columns).
* **Missing Files**:
  * `ground_truth_10000.csv` (not present; we will verify if ground truth features are already embedded within `synthetic_execution_logs_10000.csv` or if a separate ground truth file is needed).
  * Saved model files (`sandisk_failure_model.pkl`, `sandisk_threshold.pkl`, `sandisk_config_features.pkl`).
  * Python source code (`src/` modules) and Streamlit app (`app.py`).
  * Python virtual environment and `requirements.txt`.
* **Status**: Clean workspace ready for structured project setup.

---

## 2. Proposed Project Directory Structure

```text
SandDisk/
│
├── data/
│   ├── synthetic_execution_logs_10000.csv
│   └── ground_truth_10000.csv  (if applicable)
│
├── models/
│   ├── sanddisk_failure_model.pkl
│   ├── sanddisk_threshold.pkl
│   └── sanddisk_config_features.pkl
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Leakage-free feature matrix preparation & ColumnTransformer
│   ├── train_model.py       # Stratified split, RandomForest training, threshold optimization on Val set
│   ├── prediction.py        # Failure risk score & risk classification (LOW/MEDIUM/HIGH)
│   ├── explanation.py       # Feature importance & local sensitivity analysis engine
│   ├── what_if.py           # Single and multi-parameter what-if simulation engine
│   ├── similarity.py        # 70/30 weighted numeric/categorical historical run finder
│   └── recommender.py       # Safe configuration recommendation given risk threshold
│
├── app.py                   # Interactive Streamlit AI Copilot multi-page / tabbed app
├── requirements.txt         # Core dependencies (pandas, scikit-learn, streamlit, joblib, matplotlib)
├── .gitignore               # Ignores venv, raw data, model binaries (if large)
└── README.md                # Project documentation and hackathon presentation guide
```

---

## 3. Phase Breakdown

### Phase 1 — Environment & Directory Setup
* Organize directory structure (`data/`, `models/`, `src/`).
* Move `synthetic_execution_logs_10000.csv` to `data/`.
* Create `requirements.txt` and install dependencies (`pandas`, `numpy`, `scikit-learn`, `streamlit`, `joblib`).
* Create `.gitignore` and `README.md`.
* Validate dataset integrity (checking target distributions, column types).

### Phase 2 — Leakage-Free ML Model Rebuild
* Recreate `src/preprocessing.py`:
  * Drop identifiers (`run_id`, `timestamp`, `random_seed`, `seed_group`, `pass_fail`, `failure_type`, `execution_log`).
  * Drop 40 auxiliary fields (`config_aux_01` to `config_aux_40`).
  * Drop post-failure symptoms (`error_count`, `timeout_count`, `recovery_events`, `data_integrity_errors`, `watchdog_events`, `reliability_score`, `performance_score`).
  * Extract 60 prefailure features ($X$) and target label ($y = \text{FAIL}$).
* Recreate `src/train_model.py`:
  * Perform 80/20 train/test split, then split train 80/20 into train/validation (Stratified).
  * Train `RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)` with `ColumnTransformer`.
  * Compute Precision-Recall curve on **Validation Set only** to pick optimal F1 threshold ($\approx 0.1867$).
  * Evaluate final model on **Untouched Test Set** and display confusion matrix & classification metrics.
  * Save artifacts into `models/`.

### Phase 3 — AI Decision Engine Modules
* Re-implement decision engine modules under `src/`:
  * `prediction.py`: Predict probability, threshold application, risk category assignment.
  * `explanation.py`: Calculate feature importance ranking and local sensitivity shifts.
  * `what_if.py`: Single and multi-attribute configuration sensitivity engine.
  * `similarity.py`: Calculate 70% numeric + 30% categorical distance to find top $N$ historical runs.
  * `recommender.py`: Candidate filtering by predicted risk $\le \text{threshold}$ and ranking by throughput.

### Phase 4 — SandDisk Streamlit Copilot UI (`app.py`)
Build interactive Streamlit app featuring 6 dedicated copilot modes:
1. **AI Configuration Copilot**: Specify workload & risk constraint $\rightarrow$ recommended config, evidence, and risk.
2. **Failure Detective**: Select run $\rightarrow$ PASS/FAIL, failure probability, top risk factors, similar runs.
3. **What Changed?**: Side-by-side comparison of 2 configurations/runs.
4. **Configuration Playground**: Interactive slider controls to preview real-time predicted risk.
5. **What-If Simulator**: Predict impact of changing specific parameter (e.g. queue depth 32 $\rightarrow$ 128).
6. **Recommended Configurations**: Search top historical runs matching risk budget $\le X\%$.

---

## 4. Verification & Validation Plan

### Automated Verification
* Run `python src/train_model.py` to verify:
  * Leakage-free feature matrix shape ($X$ prefailure features).
  * Stratified split sample counts (Train / Val / Test).
  * Threshold selection on Val set.
  * Test set metrics (Target Accuracy $\approx 82.5\%$, FAIL Recall $\approx 69.3\%$).
  * Preservation of model pickles in `models/`.
* Test all decision engine modules with unit verification scripts.

### Manual Verification
* Run `streamlit run app.py` and interactively test each section of the Copilot interface.

---

## User Review Required

> [!IMPORTANT]
> **Dataset Row Count Note**: The workspace dataset `synthetic_execution_logs_10000.csv` has 4,939 rows. In your Colab run, a subset of 4,389 rows was used. During Phase 2, we can either use the full 4,939 validated rows or filter to the exact 4,389 subset if desired. The methodology and stratified ratio remain identical.
