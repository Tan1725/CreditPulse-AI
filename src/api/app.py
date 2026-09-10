"""
CreditPulse-AI Production Underwriting Microservice
FastAPI application providing low-latency credit decisioning,
SHAP adverse action attribution, DuckDB SQL analytics, and drift auditing.
"""

import time
import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ApplicantFeatures, UnderwritingDecisionResponse,
    RiskFactor, SqlQueryRequest, SqlQueryResponse
)
from src.database.query_runner import execute_custom_query
from src.ml.explainability import explain_single_applicant
from src.ml.drift import evaluate_portfolio_drift, generate_macro_shock_batch

app = FastAPI(
    title="CreditPulse-AI Underwriting API",
    description="Enterprise Credit Risk Underwriting & Financial Decisioning Microservice",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Global state for loaded models and metadata
STATE = {
    "model": None,
    "threshold": 0.20,
    "metadata": None,
    "threshold_config": None
}

def load_artifacts():
    """Load model, metadata, and threshold configuration into memory."""
    try:
        model_path = MODELS_DIR / "calibrated_model.joblib"
        meta_path = MODELS_DIR / "feature_metadata.json"
        thresh_path = MODELS_DIR / "threshold_optimization.json"

        if model_path.exists():
            STATE["model"] = joblib.load(model_path)
        if meta_path.exists():
            with open(meta_path, "r") as f:
                STATE["metadata"] = json.load(f)
        if thresh_path.exists():
            with open(thresh_path, "r") as f:
                cfg = json.load(f)
                STATE["threshold_config"] = cfg
                STATE["threshold"] = cfg.get("optimal_threshold", 0.20)
    except Exception as e:
        print(f"Warning: Error loading artifacts: {e}")

@app.on_event("startup")
def startup_event():
    load_artifacts()

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "CreditPulse-AI Underwriting Microservice",
        "status": "online",
        "optimal_threshold": STATE["threshold"],
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health"])
def health_check():
    model_ready = STATE["model"] is not None
    return {
        "status": "healthy" if model_ready else "initializing",
        "model_loaded": model_ready,
        "optimal_decision_threshold": STATE["threshold"]
    }

@app.post("/predict", response_model=UnderwritingDecisionResponse, tags=["Underwriting"])
def predict_applicant(applicant: ApplicantFeatures):
    """
    Sub-15ms real-time loan underwriting endpoint.
    Applies calibrated machine learning, evaluates cost-sensitive profit matrix,
    and returns ECOA/FCRA-compliant adverse action factors.
    """
    if STATE["model"] is None:
        load_artifacts()
        if STATE["model"] is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifacts not yet generated. Please run run_pipeline.py first."
            )

    start_t = time.perf_counter()
    app_dict = applicant.model_dump()

    # Convert applicant to DataFrame for inference
    df_single = pd.DataFrame([app_dict])
    
    # Predict calibrated probability of default
    prob_default = float(STATE["model"].predict_proba(df_single)[0, 1])
    optimal_tau = STATE["threshold"]

    # Decisioning Logic
    # If prob < optimal_tau: Approve
    # If between tau and tau + 0.08: Manual Underwriter Review
    # If > tau + 0.08: Decline
    review_band = 0.06
    if prob_default < optimal_tau:
        decision = "APPROVE"
    elif prob_default < (optimal_tau + review_band):
        decision = "REFER_MANUAL_REVIEW"
    else:
        decision = "DECLINE"

    # Risk Tiering
    if prob_default < 0.05:
        risk_tier = "Tier 1: Prime (Ultra Low Risk)"
    elif prob_default < 0.12:
        risk_tier = "Tier 2: Near-Prime (Moderate Risk)"
    elif prob_default < 0.25:
        risk_tier = "Tier 3: Subprime (High Risk)"
    else:
        risk_tier = "Tier 4: Deep Subprime (Severe Risk)"

    # Expected Profit calculation
    # Expected Return = (1 - p_default) * Interest Profit - p_default * Loss Given Default
    interest_rev = applicant.loan_amount * (applicant.interest_rate / 100.0) * (applicant.term_months / 12.0) * 0.80
    default_loss = applicant.loan_amount * 0.70
    expected_profit = round(float((1.0 - prob_default) * interest_rev - prob_default * default_loss), 2)

    # Adverse Action SHAP Factors
    raw_reasons = explain_single_applicant(app_dict)
    top_3_drivers = [
        RiskFactor(
            feature=r["feature"],
            shap_value=r["shap_value"],
            impact=r["impact"]
        ) for r in raw_reasons[:3]
    ]

    latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

    return UnderwritingDecisionResponse(
        decision=decision,
        default_probability=round(prob_default, 4),
        optimal_threshold=optimal_tau,
        risk_tier=risk_tier,
        expected_net_profit=expected_profit,
        top_risk_drivers=top_3_drivers,
        inference_latency_ms=latency_ms
    )

@app.post("/sql/query", response_model=SqlQueryResponse, tags=["Analytics"])
def query_warehouse(req: SqlQueryRequest):
    """
    Executes read-only SQL queries directly against the DuckDB analytical warehouse.
    """
    # Basic safety check
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    if any(token in req.query.upper() for token in forbidden):
        raise HTTPException(status_code=400, detail="Only read-only SELECT queries are permitted.")

    try:
        df = execute_custom_query(req.query)
        return SqlQueryResponse(
            columns=list(df.columns),
            row_count=len(df),
            data=df.head(100).to_dict(orient="records")
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/audit/drift", tags=["Governance"])
def audit_drift(simulate_shock_severity: float = 0.0):
    """
    Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov statistics.
    Optional simulate_shock_severity (0.0 to 1.0) injects macroeconomic stress to test alarms.
    """
    test_path = MODELS_DIR / "holdout_test.parquet"
    if not test_path.exists():
        raise HTTPException(status_code=404, detail="Holdout test data not found.")

    df_base = pd.read_parquet(test_path)
    if simulate_shock_severity > 0.0:
        df_curr = generate_macro_shock_batch(df_base, shock_severity=simulate_shock_severity)
    else:
        # Split holdout in half to test baseline stability
        df_curr = df_base.sample(frac=0.5, random_state=99)

    num_features = [
        "fico_score", "revolving_utilization_ratio",
        "total_debt_to_income_ratio", "annual_income", "rolling_avg_dpd_6m"
    ]
    report = evaluate_portfolio_drift(df_base, df_curr, num_features)
    return report
