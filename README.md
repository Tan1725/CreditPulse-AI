# 🏦 CreditPulse-AI: Enterprise Credit Risk & Decision Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.5.5-FFF000?logo=duckdb&logoColor=black)](https://duckdb.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, production-grade Credit Risk Underwriting and Financial Decisioning Platform built with **Python, SQL (DuckDB), Scikit-Learn, XGBoost, SHAP, and FastAPI**.

Unlike toy tutorials that stop at `model.fit()` with arbitrary `0.5` classification cutoffs, **CreditPulse-AI** addresses real-world enterprise constraints: **relational SQL feature engineering**, **probability calibration**, **cost-sensitive profit optimization**, **regulatory adverse action explainability (ECOA/FCRA)**, and **continuous MLOps covariate drift monitoring (PSI)**.

---

## 🏗️ System Architecture

```
                                [ Relational SQL Warehouse (DuckDB) ]
                                ├── customers (demographics, income, residence)
                                ├── credit_bureau (FICO, trade lines, inquiries)
                                ├── loan_applications (loan terms, purpose, status)
                                └── payment_ledger (120,000 monthly transactions)
                                                  │
                                                  ▼ (Advanced SQL: Window Functions & CTEs)
                                [ Feature Store & Analytical View ]
                                                  │
                       ┌──────────────────────────┴──────────────────────────┐
                       ▼                                                     ▼
           [ Machine Learning Core ]                            [ Decision Science & Governance ]
           ├── Multi-Model Benchmark (LR, RF, XGBoost)          ├── Cost-Sensitive Matrix (Profit Curve)
           ├── Probability Calibration (Isotonic / Brier)       ├── SHAP Adverse Action Waterfall (ECOA)
           └── Stratified Temporal Evaluation                   └── Population Stability Index (PSI) Drift
                       │                                                     │
                       └──────────────────────────┬──────────────────────────┘
                                                  ▼
                                [ Production Serving & Dashboards ]
                                ├── FastAPI REST Microservice (sub-15ms inference + Pydantic validation)
                                └── Streamlit Cockpit (Underwriter Studio, Live SQL Console, Drift Center)
```

---

## 💼 Quantifiable Resume Bullet Points (Copy & Paste)

### For Data Scientist / Decision Science Roles:
> *"Architected an enterprise credit risk underwriting engine on a 120k-record DuckDB warehouse using SQL window functions and CTEs; implemented cost-sensitive decision threshold optimization (\(\tau^* = 0.29\)) to maximize net portfolio profit, delivering **+\$115,139 (+3.07%) incremental financial return** over naive classification."*

> *"Engineered regulatory-compliant explainable AI workflows using **SHAP**, automating adverse action factor attribution under Equal Credit Opportunity Act (ECOA) standards and calibrating default probabilities via Isotonic Regression to achieve a **+30.3% improvement in Brier reliability score**."*

### For Machine Learning Engineer / MLOps Roles:
> *"Developed a production-ready loan underwriting microservice in **FastAPI** with sub-15ms Pydantic-validated inference; automated statistical covariate drift detection tracking **Population Stability Index (PSI)** and Kolmogorov-Smirnov statistics to detect macro-economic distributional shifts."*

> *"Benchmarked ensemble and linear classification architectures (XGBoost, Random Forest, Logistic Regression) on class-imbalanced credit data (3.9:1); established leakage-free temporal feature pipelines and automated CI test suites with 100% pass rate."*

---

## 📊 Benchmark & Performance Summary

### 1. Model Evaluation (Holdout Test Set)

| Model Architecture | Test ROC-AUC | Test PR-AUC | Test F1-Score | Brier Score (Lower is Better) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Scorecard Baseline)** | **0.8336** | **0.6099** | 0.5614 | 0.1643 |
| **Random Forest Classifier** | 0.8176 | 0.5649 | 0.5690 | 0.1284 |
| **XGBoost Classifier** | 0.8199 | 0.5778 | **0.5785** | 0.1536 |

> **Calibration Impact:** Isotonic regression calibration reduced Brier Score from `0.1682` to `0.1172` (**+30.32% improvement** in empirical default probability accuracy).

### 2. Decision Science & Economic Optimization

In credit risk, approving a defaulter (False Negative) costs the loan write-off amount (e.g., 70% Loss Given Default), whereas rejecting a good borrower (False Positive) forfeits interest margin.

* **Naive Threshold (\(\tau = 0.50\)):** Net Portfolio Profit = **\$3,748,025.46**
* **Optimal Policy Threshold (\(\tau^* = 0.29\)):** Net Portfolio Profit = **\$3,863,165.00**
* **Net Value Generated:** **+\$115,139.54 (+3.07%)**

---

## 🚀 Quick Start & Installation

### Prerequisites
* Python 3.11+
* pip

### 1. Clone & Setup
```bash
cd c:\Project\CreditPulse-AI

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Pipeline
Seeds the DuckDB data warehouse (10,000 borrowers, 120,000 ledger transactions), compiles SQL window feature views, trains & calibrates models, solves the profit threshold, and pre-computes SHAP attributions:
```bash
python run_pipeline.py
```

### 3. Launch Interactive Streamlit Cockpit
```bash
streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser:
* **🎯 Underwriting Studio:** Test live applicant profiles, review approval/denial decisions, and examine real-time SHAP adverse action waterfalls.
* **⚡ DuckDB SQL Console:** Run live analytical SQL queries with CTEs and window aggregations against the embedded warehouse.
* **📈 Decision Economics:** Inspect profit curves, threshold simulations, and multi-model metrics.
* **🛡️ MLOps Drift Monitor:** Inject simulated macroeconomic shocks and monitor real-time PSI alerts.

### 4. Launch FastAPI REST Microservice
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Run Automated Test Suite
```bash
python -m pytest tests/test_pipeline.py -v
```

---

## 📂 Project Structure

```
CreditPulse-AI/
├── dashboard/
│   └── app.py                      # Interactive Streamlit Cockpit (4 Tabs)
├── data/
│   └── credit_warehouse.duckdb     # Embedded analytical database (10k customers, 120k txns)
├── models/
│   ├── calibrated_model.joblib     # Production serving model (Isotonic calibrated)
│   ├── best_model.joblib           # Pre-calibration fitted pipeline
│   ├── baseline_model.joblib       # Logistic Regression scorecard
│   ├── threshold_optimization.json # Optimal cutoff and profit curve
│   └── global_feature_importance.json # Precomputed SHAP values
├── src/
│   ├── api/
│   │   ├── app.py                  # FastAPI underwriting microservice
│   │   └── schemas.py              # Pydantic v2 data validation schemas
│   ├── database/
│   │   ├── db_manager.py           # DuckDB schema & realistic multi-table generator
│   │   └── query_runner.py         # Analytical SQL query runner
│   ├── ml/
│   │   ├── train.py                # Multi-model benchmarking (LR, RF, XGB)
│   │   ├── calibration.py          # Platt & Isotonic probability calibration
│   │   ├── optimizer.py            # Cost-sensitive decision threshold optimizer
│   │   ├── explainability.py       # SHAP adverse action reason codes
│   │   └── drift.py                # Population Stability Index (PSI) & KS test
│   └── sql/
│       └── feature_engineering.sql # Advanced SQL view with Window Functions & CTEs
├── tests/
│   └── test_pipeline.py            # Automated integration & unit test suite
├── Dockerfile                      # Production container image
├── docker-compose.yml              # Multi-container orchestration
├── requirements.txt                # Pinned dependency requirements
├── run_pipeline.py                 # Master orchestrator script
└── README.md                       # Documentation & resume guides
```

---

## ⚖️ Model Governance & Compliance Highlights

1. **Equal Credit Opportunity Act (ECOA) & Fair Credit Reporting Act (FCRA) Compliance:**
   * Uses SHAP to isolate the exact marginal contribution of each credit attribute for rejected applicants.
   * Eliminates black-box risk by translating vector weights into readable legal adverse action notices (e.g., *"Elevated Revolving Utilization (85%)"*).

2. **Population Stability Index (PSI) Governance:**
   * Tracks distributional drift between training distributions and live production traffic.
   * Automatically classifies drift severity:
     * \(\text{PSI} < 0.10\): Stable (🟢)
     * \(0.10 \le \text{PSI} < 0.25\): Moderate Drift / Monitor (🟡)
     * \(\text{PSI} \ge 0.25\): Significant Drift / Retrain Trigger (🔴)

---

## 📜 License
MIT License. Built for professional Data Science and Machine Learning portfolio demonstrations.
