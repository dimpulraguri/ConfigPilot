# SandDisk — AI Configuration & Reliability Copilot

**SandDisk** is an AI-powered reliability and configuration decision engine designed for complex computing and storage systems. It enables systems engineers to predict failure risks, understand feature sensitivity, compare system runs, perform What-If parameter simulations, search historical execution logs, and recommend safe, high-performance configurations under strict failure-risk budgets.

> **Hackathon Product Principle**: *"Don't just monitor failures. Help engineers choose safer configurations before failures happen."*

---

## 🌟 Core Decision Engine Capabilities

1. **AI Configuration Copilot**: Specify workload requirements and max acceptable failure risk to get top recommended settings backed by historical evidence.
2. **Failure Detective**: Inspect historical execution runs, examine percentile-based telemetry observations, and analyze local sensitivity.
3. **What Changed?**: Side-by-side run comparison identifying parameter differences and risk divergence.
4. **Configuration Playground**: Interactive parameter sliders to preview real-time failure probability predictions.
5. **What-If Simulator**: Single-parameter sweeps and multi-parameter hypothetical scenario simulation.
6. **Safe Recommender**: Ranks historical candidate configurations by throughput while enforcing predicted risk constraints ($\le X\%$).

---

## 🛡️ Machine Learning Methodology & Leakage Prevention

To ensure true predictive validity and avoid artificial high accuracy through data leakage:
* **Target Label**: $y = 1$ if `pass_fail == "FAIL"` else $0$.
* **Excluded Metadata**: `run_id`, `timestamp`, `random_seed`, `seed_group`, `pass_fail`, `failure_type`, `execution_log`.
* **Excluded Auxiliary Fields**: `config_aux_01` to `config_aux_40`.
* **Excluded Outcome Symptoms**: `error_count`, `timeout_count`, `recovery_events`, `data_integrity_errors`, `watchdog_events`, `reliability_score`, `performance_score`.
* **Model**: `RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")`.
* **Threshold Optimization**: Optimal threshold ($\approx 18.67\%$) selected exclusively on **Validation Set F1 Max**.
* **Untouched Test Evaluation**: Final performance evaluated only on untouched test set ($\text{FAIL Recall} \approx 69.3\%$, $\text{Accuracy} \approx 82.5\%$).

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Virtual Environment Setup
Open a terminal in VS Code and run:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

### 2. Train Model & Generate Decision Artifacts
To train the Random Forest pipeline and generate saved artifacts (`models/*.pkl`):
```bash
python src/train_model.py
```

### 3. Launch Interactive Streamlit Copilot App
```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```text
SandDisk/
│
├── data/
│   └── synthetic_execution_logs_10000.csv   # Validated synthetic log dataset
│
├── models/                                  # Saved model pickle artifacts
│   ├── sanddisk_failure_model.pkl
│   ├── sanddisk_threshold.pkl
│   └── sanddisk_config_features.pkl
│
├── src/                                     # AI Decision Engine modules
│   ├── preprocessing.py                     # Leakage-free feature matrix & ColumnTransformer
│   ├── train_model.py                       # Training pipeline & threshold optimization
│   ├── prediction.py                        # Failure risk calculation & classification
│   ├── explanation.py                       # Feature importance & local sensitivity
│   ├── what_if.py                           # Single & multi-parameter What-If simulator
│   ├── similarity.py                        # 70/30 weighted distance historical search
│   └── recommender.py                       # Risk-constrained high-throughput recommendations
│
├── app.py                                   # Streamlit Copilot application
├── requirements.txt                         # Python package dependencies
├── .gitignore                               # Version control ignore configuration
└── README.md                                # Documentation
```

---

## ⚠️ Limitations & Disclaimer

This project is a **synthetic-data prototype**. 
- Feature importance and local sensitivity reflect **model reliance and numerical sensitivity**, not physical causal mechanics.
- What-If simulations evaluate model predictions under altered inputs holding telemetry constant.
- Recommendations represent top historical candidates under model risk constraints, not guaranteed real-world deployment outcomes.
