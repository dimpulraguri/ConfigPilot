⚙️ ConfigPilot: AI Configuration & Reliability Copilot
Predict. Explain. Simulate. Recommend.

An AI-powered configuration and reliability intelligence platform that helps engineers understand execution failures, identify influential configuration parameters, investigate failure patterns, evaluate configuration changes, and discover lower-predicted-risk configurations from historical execution data.

Built for the SandDisk Challenge

**Core Product Principle**

> "Don't just monitor failures. Help engineers choose safer configurations before failures happen."

Python Scikit-learn Streamlit Pandas NumPy

🔗 Live Links
Live Application: Add Streamlit URL after deployment
GitHub Repository: Add GitHub repository URL
Demo Video: Add demo video URL

📑 Table of Contents
1.Overview
2.Challenge Requirements Compliance
3.Screenshots
4.Core Features ⭐
5.AI/ML Capabilities ⭐
6.Technology Stack ⭐
7.Project Architecture
8.Project Structure
9.Dataset & Methodology
10.Installation & Setup ⭐
11.Model Evaluation
12.Explainability & Reliability Intelligence
13.Limitations & Responsible Interpretation
14.Future Enhancements
15.Contributors
16.License
17.Acknowledgements

📖 Overview
ConfigPilot is an AI Configuration & Reliability Copilot designed to help engineers move beyond simply observing execution failures toward understanding, predicting, investigating, and reducing configuration-related risk.

Modern execution environments can contain a large number of configuration parameters, workload conditions, telemetry signals, and randomized variables. When an execution fails, identifying what influenced the failure, what changed, whether the behavior is repeatable, and which configuration should be tried next can be difficult.

ConfigPilot addresses this workflow through an integrated analytics and AI/ML platform.

The system transforms execution data into actionable engineering insights through:
- Configuration intelligence
- Failure-risk prediction
- Failure investigation
- Configuration comparison
- Randomization analysis
- What-if configuration analysis
- Historical configuration recommendations
- Natural-language interaction through Ask ConfigPilot

The platform follows a simple workflow:

```
Execution Data
      ↓
Data Processing & Feature Extraction
      ↓
AI / ML Reliability Engine
      ↓
Prediction + Explainability + Analytics
      ↓
What-If Analysis
      ↓
Configuration Recommendations
      ↓
Engineering Decision
```

✅ Challenge Requirements Compliance
ConfigPilot addresses the major analytical requirements of the challenge through an integrated reliability intelligence workflow.

| Requirement | Status | Implementation |
|---|---|---|
| Configuration analysis | ✅ | Configuration Intelligence |
| Failure prediction | ✅ | Random Forest risk prediction |
| Failure factor analysis | ✅ | Feature importance + sensitivity analysis |
| Configuration comparison | ✅ | What Changed? |
| Randomization analysis | ✅ | Randomization & Determinism |
| Failure investigation | ✅ | Failure Detective |
| Configuration recommendations | ✅ | Risk-constrained historical recommender |
| What-if analysis | ✅ | Configuration risk estimator |
| Interactive dashboard | ✅ | Streamlit application |
| Natural-language querying | ✅ | Ask ConfigPilot |
| Explainable AI | ✅ | Feature importance + local sensitivity |
| Automated insights | ✅ | Integrated analytics and explanations |
| Documentation | ✅ | README + methodology documentation |

📊 Dataset
The current prototype uses organizer-approved synthetic execution data.

The challenge organizers instructed participants to generate the dataset using an LLM because the underlying execution data could not be distributed due to NDA and infrastructure constraints.

Therefore, ConfigPilot uses synthetic execution records to demonstrate the complete analytical and AI/ML pipeline.

**Current Dataset**

| Property | Value |
|---|---|
| Execution records | 4,939 |
| Total columns | 114 |
| PASS executions | 4,289 |
| FAIL executions | 650 |
| PASS rate | 86.8% |
| FAIL rate | 13.2% |
| Missing values | 0 |
| Duplicate run IDs | 0 |
| Predictive features | 60 |
| Numeric features | 48 |
| Categorical features | 12 |

The dataset contains configuration parameters, workload characteristics, telemetry, randomized variables, execution outcomes, and failure-related information.

📸 Screenshots
Add screenshots of the deployed application here after deployment.

🏠 Dashboard Overview
The dashboard provides an executive-level summary of the execution dataset, model performance, and available reliability intelligence modules.

[ Add Dashboard Screenshot ]

🧠 Configuration Intelligence
Ranks configuration and telemetry features according to their contribution to the predictive model.

[ Add Configuration Intelligence Screenshot ]

🎯 Failure Risk Prediction
Allows engineers to evaluate an execution configuration and obtain:

- Failure probability
- PASS/FAIL prediction
- Risk level
- Influential factors

[ Add Failure Risk Screenshot ]

🔎 Failure Detective
Provides a deeper investigation view for execution failures, including telemetry, prediction results, sensitivity information, and contributing factors.

[ Add Failure Detective Screenshot ]

🔄 What Changed?
Compares successful and failed executions to identify differences in:

- Configuration
- Telemetry
- Risk
- Execution behavior
- Logs

[ Add What Changed Screenshot ]

🎛️ What-If Analysis
Allows engineers to modify configuration parameters and observe how the model's predicted failure risk changes.

[ Add What-If Screenshot ]

💡 Configuration Recommendations
Finds historical configurations that satisfy a user-defined predicted-risk constraint and ranks them using observed historical performance.

[ Add Recommendation Screenshot ]

🚀 Core Features

**1. 🧠 Configuration Intelligence**

Identifies parameters the trained model relies on most strongly when distinguishing PASS and FAIL executions.

The system provides:
- Ranked feature importance
- Feature-level observations
- High/low percentile indicators
- Local sensitivity analysis
- Engineering-oriented explanations

> Important: Model feature importance indicates predictive reliance. It does not prove that a feature is causally responsible for failures.

**2. 🎯 Failure Risk Prediction**

ConfigPilot uses a machine-learning pipeline to estimate the probability of execution failure.

For a given configuration, the system provides:

```
Failure Probability
        ↓
Risk Classification
        ↓
PASS / FAIL Prediction
        ↓
Explanation
```

The prediction engine uses a Random Forest classifier with 300 trees.

**3. 🔎 Failure Detective**

Failure Detective provides an investigation-oriented view of an execution.

It combines:
- Predicted failure probability
- Actual execution outcome
- Telemetry
- Feature observations
- Local sensitivity
- Influential factors

This helps engineers move from:

"The execution failed."

toward:

"What signals and configuration characteristics were associated with this failure?"

**4. 🔄 What Changed?**

What Changed? compares successful and failed executions to identify meaningful differences.

The comparison can include:
- Configuration parameters
- Telemetry metrics
- Risk score
- Execution logs
- Configuration deltas

This supports faster debugging and failure investigation.

**5. 🎛️ What-If Configuration Analysis**

Engineers can modify selected configuration parameters and evaluate the resulting predicted failure risk.

Supported configuration dimensions include:
- Workload type
- Traffic intensity
- Cache size
- Cache policy
- Memory allocation
- Memory policy
- Thread count
- CPU cores
- Queue depth
- I/O parallelism
- Request size
- I/O scheduler
- Read/write ratio

Example workflow:

```
Current Configuration
        ↓
Change Parameter
        ↓
Run Model Prediction
        ↓
Compare Predicted Risk
        ↓
Evaluate Configuration
```

> Important: The current what-if module is a configuration-risk estimator, not a physics-based performance simulator.

**6. 🎲 Randomization & Determinism Analysis**

ConfigPilot analyzes randomized execution variables to investigate whether failures appear associated with particular randomization conditions.

The module provides:
- Randomization feature impact ranking
- Seed-group failure analysis
- Failure-rate comparisons
- Repeatability-oriented analysis

The prototype also explicitly reports when the available data is insufficient to establish deterministic behavior reliably.

**7. 💡 Configuration Recommendations**

The recommendation engine allows engineers to specify a maximum acceptable predicted failure risk.

The system then:

```
Risk Constraint
      ↓
Historical Configurations
      ↓
Predict Risk
      ↓
Filter Candidates
      ↓
Rank by Historical Performance
      ↓
Recommend Candidate
```

The recommendation is presented as:

"Top historical candidate under the selected predicted-risk constraint."

Recommendations are not presented as guarantees.

**8. 🤖 Ask ConfigPilot**

Ask ConfigPilot provides a natural-language interface for interacting with the platform.

Example questions:
- What configuration settings influence failure the most?
- Which configurations have the best performance?
- Which randomization parameters matter most?
- Are failures deterministic or random?
- Can future failures be predicted?
- Recommend a configuration below 5% predicted failure risk.
- What does the 29% threshold mean?
- Is this a physics simulator?

The current implementation uses a deterministic intent engine that maps natural-language questions to the platform's existing analytical functions.

This keeps the prototype functional without requiring an external LLM API at runtime.

🤖 AI/ML Capabilities
ConfigPilot combines several analytical techniques into one reliability workflow.

**Machine Learning**
- Random Forest classification
- Probability-based failure prediction
- Class balancing
- Validation-based threshold selection

**Explainable AI**
- Global feature importance
- Percentile-based observations
- Local sensitivity analysis

**Statistical Analysis**
- Failure-rate analysis
- Feature comparisons
- Randomization impact analysis
- Configuration comparisons

**Recommendation Intelligence**
- Predicted-risk filtering
- Historical configuration search
- Performance-based ranking

**Natural Language Interaction**
- Intent detection
- Query routing
- Human-readable analytical responses

🛠 Technology Stack

**Programming & Data**

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Joblib | Model persistence |

**Machine Learning**

| Component | Implementation |
|---|---|
| Model | Random Forest Classifier |
| Trees | 300 |
| Class balancing | Balanced class weights |
| Categorical encoding | One-Hot Encoding |
| Threshold selection | Validation-set F1 optimization |
| Prediction output | Failure probability + risk classification |

**Application**

| Technology | Purpose |
|---|---|
| Streamlit | Interactive web dashboard |
| Python modules | Modular backend architecture |
| Git | Version control |
| GitHub | Source-code hosting |

🏗 Project Architecture
ConfigPilot follows a modular architecture separating data processing, machine learning, explainability, analysis, recommendations, and user interaction.

```
                 ┌──────────────────────────┐
                 │   Execution Data Source  │
                 │ CSV / Exported Logs /    │
                 │ Simulator / Collection   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Data Ingestion &          │
                 │ Validation                │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Feature Extraction &      │
                 │ Normalization             │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ AI / ML Reliability      │
                 │ Engine                   │
                 │                          │
                 │ Random Forest            │
                 │ Risk Prediction          │
                 └────────────┬─────────────┘
                              │
             ┌────────────────┼─────────────────┐
             ▼                ▼                 ▼
      ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
      │Explainability│  │ Reliability │  │ Configuration│
      │& Sensitivity │  │ Analytics   │  │ Intelligence │
      └──────┬──────┘  └──────┬──────┘  └──────┬───────┘
             │                │                 │
             └────────────────┼─────────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ What-If & Recommendation │
                 │ Engine                   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     ConfigPilot UI       │
                 │ Dashboard + AI Copilot   │
                 └──────────────────────────┘
```

**Production Deployment Path**

The prototype is designed so that the synthetic dataset can be replaced by an organization-permitted execution-data source.

```
Organization Execution Environment
              ↓
Log / Data Export or Collection Layer
              ↓
ConfigPilot Data Ingestion
              ↓
Validation & Feature Extraction
              ↓
Organization-Specific Model Training
              ↓
Risk Prediction & Explainability
              ↓
Recommendations & Engineering Insights
```

The production architecture does not depend on the challenge organizers supplying real-time data.

📂 Project Structure
The project follows a modular structure to keep the codebase organized, maintainable, and scalable.

```
ConfigPilot/
│
├── data/
│   └── synthetic_execution_logs_10000.csv
│
├── models/
│   ├── sanddisk_config_features.pkl
│   ├── sanddisk_failure_model.pkl
│   └── sanddisk_threshold.pkl
│
├── src/
│   ├── __init__.py
│   ├── chatbot.py
│   ├── explanation.py
│   ├── prediction.py
│   ├── preprocessing.py
│   ├── recommender.py
│   ├── similarity.py
│   ├── train_model.py
│   ├── what_if.py
│   ├── test_engine.py
│   └── test_full_pipeline.py
│
├── app.py
├── requirements.txt
├── README.md
├── implementation_plan.md
└── .gitignore
```

**Module Responsibilities**

| Module | Purpose |
|---|---|
| app.py | Streamlit application and UI |
| preprocessing.py | Feature preparation and preprocessing |
| train_model.py | Model training pipeline |
| prediction.py | Failure-risk prediction |
| explanation.py | Feature importance and sensitivity |
| what_if.py | Configuration what-if analysis |
| similarity.py | Configuration similarity |
| recommender.py | Risk-constrained recommendations |
| chatbot.py | Ask ConfigPilot intent handling |

📊 Dataset & Methodology

**Feature Selection**

The model uses pre-failure / configuration-relevant features while explicitly removing fields that could leak the outcome.

Excluded fields include:
- run_id
- timestamp
- random_seed
- seed_group
- pass_fail
- failure_type
- execution_log
- Auxiliary configuration fields
- Post-outcome failure indicators

Post-outcome fields such as:
- error_count
- timeout_count
- recovery_events
- data_integrity_errors
- watchdog_events
- reliability_score
- performance_score

are excluded from prediction to reduce target leakage.

**Train / Validation / Test Split**

The dataset is divided using stratified sampling.

| Split | Samples |
|---|---|
| Training | 3,160 |
| Validation | 791 |
| Test | 988 |

The validation set is used for threshold selection, while the test set remains untouched during threshold optimization.

⚙️ Installation & Setup Instructions
Follow the steps below to run ConfigPilot locally.

**Prerequisites**

Make sure the following software is installed:
- Python 3.x
- pip
- Git

**Clone the Repository**

```
git clone <GitHub repository URL>
cd ConfigPilot
```

**Install Dependencies**

```
pip install -r requirements.txt
```

**Train the Model** (if model artifacts are not already present in `/models`)

```
python src/train_model.py
```

**Run the Application**

```
streamlit run app.py
```

Open your browser and visit the local URL shown in the terminal (typically `http://localhost:8501`).

**Run Tests**

```
python src/test_engine.py
python src/test_full_pipeline.py
```

🎯 Model Evaluation
The prototype uses a Random Forest classifier with 300 trees.

The decision threshold was selected on the validation set using F1 maximization, resulting in:

**Decision threshold: 29.00%**

**Held-Out Test Performance**

| Metric | Result |
|---|---|
| Accuracy | 83.3% |
| Macro F1 | 0.706 |
| FAIL Recall | 66.92% |
| Test failures detected | 87 / 130 |

**Confusion Matrix**

```
                    Predicted
                 PASS       FAIL

Actual PASS       736        122
Actual FAIL        43         87
```

The prototype emphasizes FAIL recall and failure-oriented metrics rather than relying only on overall accuracy.

> Important: These metrics are measured on the current synthetic dataset and should not be interpreted as real-world production performance.

🔍 Explainability & Reliability Intelligence
ConfigPilot does not treat a prediction as a black-box answer.

The platform combines:

```
Prediction
    +
Feature Importance
    +
Input Observation
    +
Local Sensitivity
    +
Historical Evidence
    =
Engineering-Oriented Explanation
```

**Global Explanation**

Shows which features the model relies on most strongly across the dataset.

**Local Explanation**

Evaluates how small changes in selected inputs affect the model's predicted failure probability.

**Observation Context**

Features are categorized according to their observed percentile:
- VERY LOW
- LOW
- NORMAL
- HIGH
- VERY HIGH

These observations provide context for the prediction but are not treated as automatic causal explanations.

⚠️ Limitations & Responsible Interpretation
ConfigPilot is currently a functional proof-of-concept, not a production-certified reliability system.

**Synthetic Dataset**

The current prototype uses organizer-approved synthetic execution data because the underlying execution data could not be distributed due to NDA and infrastructure constraints.

**Feature Importance ≠ Causality**

A highly important feature means the model relies on it for prediction. It does not prove that changing that feature alone causes a failure.

**What-If ≠ Physics Simulation**

The current what-if engine estimates how configuration changes affect predicted failure risk.

It does not model physical system behavior or guarantee changes in actual throughput, latency, or hardware performance.

**Recommendations ≠ Guarantees**

Recommended configurations are historical candidates satisfying the selected predicted-risk constraint.

They are not guaranteed to be globally optimal or physically safe.

**Determinism Analysis**

The current dataset does not contain enough exact repeated 13-parameter trials to establish deterministic failure behavior reliably.

Therefore, ConfigPilot reports the limitation rather than making an unsupported deterministic/random conclusion.

🚀 Future Enhancements
ConfigPilot is designed to evolve from a synthetic-data prototype into an organization-specific reliability intelligence platform.

- 🔌 **Organization Data Integration** – Connect to permitted execution-log exports, test environments, simulators, or existing data infrastructure.
- 🧠 **Organization-Specific Model Training** – Retrain and calibrate models using organization-specific historical execution data.
- 📈 **Time-Series Failure Prediction** – Add temporal models for detecting early degradation and predicting failures before execution completion.
- 🔬 **Advanced Explainable AI** – Integrate advanced explainability techniques such as SHAP-based local and global explanations.
- 🔗 **Causal Analysis** – Introduce causal inference techniques to distinguish predictive correlation from actual configuration effects.
- 🧪 **Experiment Tracking** – Track configuration experiments and compare predicted versus observed outcomes over time.
- 📊 **Pareto Optimization** – Expand recommendation logic to optimize multiple objectives such as:

```
Failure Risk
      ↕
Throughput
      ↕
Latency
      ↕
Resource Utilization
```

- 🔄 **Continuous Monitoring** – Support model monitoring, drift detection, recalibration, and automated retraining pipelines.
- 🤖 **Advanced AI Copilot** – Extend Ask ConfigPilot with an LLM-based engineering assistant while retaining deterministic analytical functions underneath.

👥 Contributors
Developed as a team project for the SandDisk Challenge.

| Contributor | Role |
|---|---|
| Add Name | AI/ML & Reliability Analytics |
| Add Name | Application & Dashboard |
| Add Name | Data Engineering |
| Add Name | Research & Documentation |

📄 License
This project was developed as a prototype submission for the SandDisk Challenge.

Add your preferred open-source license here if required by your team or challenge rules.

🙏 Acknowledgements
Special thanks to the challenge organizers for providing the problem context and permitting participants to generate synthetic datasets for prototyping under the stated NDA and infrastructure constraints.

We also acknowledge the open-source technologies that made this prototype possible:

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Git
- GitHub

⚙️ ConfigPilot: AI Configuration & Reliability Copilot
Predict. Explain. Simulate. Recommend.

Built for the SandDisk Challenge
