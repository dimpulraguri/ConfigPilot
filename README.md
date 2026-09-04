# ⚙️ ConfigPilot: AI Configuration & Reliability Copilot

### Predict. Explain. Simulate. Recommend.

> **An AI-powered configuration intelligence and reliability platform that helps engineers understand failures, evaluate configuration risk, investigate execution behavior, and identify lower-predicted-risk configurations.**

**Built for the SandDisk Challenge**

---

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![Status](https://img.shields.io/badge/Status-Prototype-success)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

## 🚀 Live Demo

🌐 **Streamlit Application:**  
> *Add deployed Streamlit URL here*

🎥 **Demo Video:**  
> *Add 5-minute demo video link here*

💻 **GitHub Repository:**  
> *Add repository URL here*

---

# 📑 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Solution](#-solution)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Screenshots](#-screenshots)
- [Dataset](#-dataset)
- [AI/ML Methodology](#-aiml-methodology)
- [Model Evaluation](#-model-evaluation)
- [Explainability](#-explainability)
- [Configuration What-If Analysis](#-configuration-what-if-analysis)
- [Recommendation Engine](#-recommendation-engine)
- [Ask ConfigPilot](#-ask-configpilot)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Demo Flow](#-demo-flow)
- [Contributors](#-contributors)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

# 🌟 Overview

Modern execution environments expose engineers to a large number of configuration parameters, workload variables, telemetry metrics, and randomized execution conditions.

When an execution fails, the difficult questions are not simply:

> **"Did it fail?"**

Engineers also need to know:

- **Why did it fail?**
- **Which configuration factors matter most?**
- **What changed between successful and failed executions?**
- **Are failures deterministic or influenced by randomization?**
- **Can the risk of a future failure be estimated?**
- **What configuration should be considered next?**

**ConfigPilot** addresses these questions through a unified AI/ML-powered reliability workflow.

### Core Product Principle

> **Don't just monitor failures. Help engineers choose lower-predicted-risk configurations before failures happen.**

ConfigPilot combines:

- Configuration intelligence
- Failure-risk prediction
- Explainable ML
- Failure investigation
- What-changed analysis
- Randomization analysis
- Configuration what-if analysis
- Historical configuration recommendations
- Natural-language querying

---

# 🎯 Problem Statement

Configuration-heavy execution systems can fail because of interactions between:

- Configuration parameters
- Workload characteristics
- Resource utilization
- Performance conditions
- Randomization variables
- Timing behavior
- Environmental conditions

Traditional monitoring systems primarily answer:

> **"What happened?"**

ConfigPilot extends this into:

> **"What happened → Why → What changed → What could happen next → What configuration should we consider?"**

---

# 💡 Solution

ConfigPilot transforms execution data into actionable reliability intelligence.

```text
Execution Data
      │
      ▼
Data Ingestion & Validation
      │
      ▼
Feature Extraction & Normalization
      │
      ▼
AI / ML Reliability Engine
      │
      ├──────────────► Failure Risk Prediction
      │
      ├──────────────► Configuration Intelligence
      │
      ├──────────────► Failure Investigation
      │
      ├──────────────► Randomization Analysis
      │
      ├──────────────► What-If Analysis
      │
      └──────────────► Configuration Recommendations
                              │
                              ▼
                    Engineer Decision Support
🧠 Key Features
1. Configuration Intelligence

Identifies configuration and execution variables that the predictive model relies on most strongly.

Provides:

Ranked feature importance
Configuration impact insights
Feature-level observations
Model-based explanations

Important: Feature importance indicates model reliance and should not be interpreted as proof of causality.

2. Failure Risk Prediction

Predicts the probability that an execution will fail.

Output
Failure probability
PASS / FAIL prediction
Risk level
Configurable decision threshold
Explainability information

The prototype uses a Random Forest classifier with 300 trees.

3. Failure Detective

Investigates individual execution failures by combining:

Prediction
Actual execution outcome
Telemetry
Configuration values
Feature sensitivity
Contributing factors

This provides engineers with a structured failure fingerprint rather than only a PASS/FAIL label.

4. What Changed?

Compares successful and failed execution scenarios.

The analysis can highlight changes in:

Configuration parameters
Telemetry
Risk score
Execution behavior

This helps answer:

"What was different when the system failed?"

5. Configuration What-If Analysis

Engineers can modify configuration parameters and observe how the model's predicted failure risk changes.

Example:

Current Configuration
        │
        ▼
Modify thread_count
        │
        ▼
Modify queue_depth
        │
        ▼
Modify cache_policy
        │
        ▼
Recalculate Predicted Risk

Note: The current implementation is a configuration-risk estimator, not a physics-based system simulator.

6. Randomization & Determinism Analysis

Analyzes randomized execution variables and seed-group behavior.

The system provides:

Randomization impact ranking
Seed-group failure-rate analysis
Repeatability analysis
Determinism assessment

The prototype explicitly reports when the available data is insufficient to establish deterministic behavior reliably.

7. Configuration Recommendations

Engineers can specify a maximum acceptable predicted failure risk.

ConfigPilot then:

Applies the risk constraint.
Evaluates historical configuration candidates.
Filters candidates exceeding the selected predicted-risk budget.
Ranks remaining candidates using observed historical performance.
Presents the strongest historical candidate.

Recommendation wording:
"Top historical candidate under the selected predicted-risk constraint."

Recommendations are decision-support suggestions, not guarantees of future success.

8. 🤖 Ask ConfigPilot

ConfigPilot provides a natural-language interface for querying the analytics engine.

Example questions:

What configuration settings influence failure the most?

Which configurations have the best performance?

Which randomization parameters matter most?

Are failures deterministic or random?

Can future failures be predicted?

Recommend a configuration below 5% predicted failure risk.

What does the 29% threshold mean?

Is this a physics simulator?

The current prototype uses a lightweight deterministic intent engine that routes questions to the corresponding analytics and ML functionality.

🏗️ Architecture
System Architecture

Architecture diagram placeholder:
Add your final architecture diagram at:

docs/images/configpilot_architecture.png

Production-Oriented Architecture
┌───────────────────────────────┐
│   Execution Environment       │
│                               │
│ Configurations + Workloads    │
│ Telemetry + Execution Logs    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Data Collection / Export      │
│ Layer                         │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ ConfigPilot Data Ingestion    │
│                               │
│ Validation                    │
│ Parsing                       │
│ Normalization                 │
│ Feature Extraction            │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ AI / ML Reliability Engine   │
│                               │
│ Failure Prediction            │
│ Feature Importance             │
│ Sensitivity Analysis           │
│ Randomization Analysis        │
└───────────────┬───────────────┘
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
   Explain   What-If   Recommend
       │        │        │
       └────────┼────────┘
                ▼
┌───────────────────────────────┐
│ ConfigPilot Dashboard         │
│                               │
│ Insights + Alerts + Actions   │
└───────────────────────────────┘
Deployment Model

The current prototype uses synthetic execution data.

In a production deployment, the ingestion layer can connect to an organization's permitted execution-data source, such as:

Existing logging infrastructure
Exported CSV/JSON execution records
Test environments
Simulation environments
Approved data pipelines

The model can then be retrained and validated using organization-specific historical execution data.

📸 Screenshots
Dashboard Overview

Screenshot placeholder: Add dashboard screenshot here.

Configuration Intelligence

Screenshot placeholder: Add feature-importance screenshot here.

Failure Risk Prediction

Screenshot placeholder: Add prediction page screenshot here.

Failure Detective

Screenshot placeholder: Add failure-analysis screenshot here.

What-If Analysis

Screenshot placeholder: Add what-if screenshot here.

Configuration Recommendations

Screenshot placeholder: Add recommendation screenshot here.

Ask ConfigPilot

Screenshot placeholder: Add copilot screenshot here.

📊 Dataset

The current prototype uses an organizer-approved synthetic execution dataset.

The challenge organizers instructed participants to generate the dataset using an LLM because the underlying execution data could not be distributed due to NDA and infrastructure constraints.

Dataset Summary
Property	Value
Execution records	4,939
Total columns	114
PASS executions	4,289
FAIL executions	650
Predictive features	60
Numeric features	48
Categorical features	12
Missing values	0
Duplicate run IDs	0
Data Split
Dataset	Samples	Failures
Training	3,160	416
Validation	791	104
Test	988	130

The split is stratified to preserve the failure ratio across datasets.

🔬 AI/ML Methodology
Feature Preparation

To avoid target leakage, the model does not use:

Run identifiers
Timestamps
Random seeds
Seed groups
Final PASS/FAIL labels
Failure types
Execution logs
Post-outcome metrics
Ground-truth auxiliary fields

The predictive feature matrix contains:

60 pre-failure features

including:

Configuration parameters
Workload characteristics
Resource metrics
Performance indicators
Environmental variables
Randomization-related inputs
Model
Random Forest Classifier
Algorithm: Random Forest
Number of Trees: 300
Class Weight: Balanced
Random State: 42
Categorical Encoding: One-Hot Encoding

The pipeline uses:

Raw Data
   │
   ▼
ColumnTransformer
   │
   ├── Numeric → Passthrough
   │
   └── Categorical → One-Hot Encoding
   │
   ▼
Random Forest
   │
   ▼
Failure Probability
📈 Model Evaluation

The decision threshold was selected using the validation set only, maximizing F1 score through precision-recall analysis.

Selected Threshold

29.00% predicted failure probability

Held-Out Test Performance
Metric	Result
Accuracy	83.3%
Macro F1	0.706
FAIL Recall	66.92%
Confusion Matrix
                 Predicted
                 PASS   FAIL
Actual PASS       736    122
Actual FAIL        43     87

The model correctly identifies 87 of 130 failures in the held-out synthetic test set.

These results demonstrate the functionality of the prototype on synthetic data and should not be interpreted as production or real-world performance.

🔍 Explainability

ConfigPilot provides both global and local explanations.

Global Explanation

Identifies which features the model relies on most strongly across the dataset.

Local Explanation

For an individual execution, the system examines:

Feature values
Percentile-based observations
Local sensitivity to feature changes

Example:

Feature
   │
   ▼
Observed Value
   │
   ▼
Dataset Percentile
   │
   ▼
Sensitivity Analysis
   │
   ▼
Human-Readable Insight

Feature importance and sensitivity are model-based signals and do not establish causal relationships.

🧪 Configuration What-If Analysis

ConfigPilot supports experimentation with controllable configuration parameters.

Current configurable parameters include:

workload_type
traffic_intensity
cache_size_mb
cache_policy
memory_allocation_mb
memory_policy
thread_count
cpu_cores
queue_depth
io_parallelism
request_size_kb
io_scheduler
read_write_ratio

The system changes selected parameters while keeping the remaining execution context fixed and evaluates the resulting predicted risk.

This enables engineers to ask:

"If I change this configuration, does the model consider the execution more or less risky?"

🏆 Recommendation Engine

The recommendation engine uses a user-defined predicted-risk constraint.

Example:

Maximum predicted failure risk
              ↓
             5%
              ↓
Historical configurations
              ↓
Risk filtering
              ↓
Performance ranking
              ↓
Top historical candidate

The recommendation is based on historical evidence available in the dataset.

It is not presented as a guaranteed optimal or guaranteed-safe configuration.

🧩 Technology Stack
Category	Technology
Language	Python
Dashboard	Streamlit
Machine Learning	Scikit-learn
Data Processing	Pandas
Numerical Computing	NumPy
Model	Random Forest
Encoding	One-Hot Encoding
Visualization	Streamlit Charts / Plotting
Version Control	Git / GitHub
Deployment	Streamlit Community Cloud
📁 Project Structure
SandDisk/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── synthetic_execution_logs_10000.csv
│
├── models/
│   ├── sanddisk_failure_model.pkl
│   ├── sanddisk_threshold.pkl
│   └── sanddisk_config_features.pkl
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── prediction.py
│   ├── explanation.py
│   ├── what_if.py
│   ├── similarity.py
│   ├── recommender.py
│   └── chatbot.py
│
└── docs/
    └── images/
        ├── configpilot_architecture.png
        ├── dashboard.png
        ├── configuration_intelligence.png
        ├── failure_prediction.png
        ├── failure_detective.png
        ├── what_if.png
        ├── recommendations.png
        └── ask_configpilot.png
⚙️ Installation & Setup
1. Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SandDisk
2. Create a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Run ConfigPilot
streamlit run app.py

The application will open in your browser.

▶️ Usage
Dashboard

Start with the Dashboard Overview to understand:

Dataset size
PASS/FAIL distribution
Model performance
Decision threshold
Available capabilities
Analyze Configuration Risk

Navigate to:

Failure Risk Prediction

Enter configuration parameters and obtain:

Failure probability
Risk level
PASS/FAIL prediction
Investigate Failures

Use:

Failure Detective

to inspect individual execution behavior.

Compare Executions

Use:

What Changed?

to identify differences between execution scenarios.

Explore Configurations

Use:

What-If Analysis

to evaluate predicted risk under modified configurations.

Get Recommendations

Use:

Configuration Recommendations

to search historical candidates under a selected predicted-risk constraint.

Ask ConfigPilot

Use natural-language questions through:

Ask ConfigPilot                    