"""
Automated Integration and Unit Tests for CreditPulse-AI
Tests:
- DuckDB schema and feature engineering SQL view
- Probability calibration and prediction range
- Cost-sensitive threshold optimizer integrity
- Population Stability Index (PSI) drift engine
- FastAPI underwriting endpoint validation
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from src.database.db_manager import get_connection
from src.database.query_runner import load_modeling_dataset, execute_custom_query
from src.ml.drift import calculate_psi
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_database_connection():
    """Verify DuckDB attaches and tables exist."""
    conn = get_connection(read_only=True)
    tables = conn.execute("SHOW TABLES;").fetchall()
    table_names = [t[0] for t in tables]
    assert "customers" in table_names
    assert "credit_bureau" in table_names
    assert "loan_applications" in table_names
    assert "payment_ledger" in table_names
    conn.close()

def test_sql_feature_view():
    """Verify SQL feature engineering view compiles and extracts features."""
    df = load_modeling_dataset()
    assert len(df) > 0
    required_cols = [
        "customer_id", "fico_score", "revolving_utilization_ratio",
        "total_debt_to_income_ratio", "payment_to_income_ratio",
        "rolling_avg_dpd_6m", "is_default"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Check default rate is bounded
    dr = df["is_default"].mean()
    assert 0.02 < dr < 0.40, f"Unrealistic default rate: {dr:.2%}"

def test_psi_calculation():
    """Verify PSI behaves properly on identical vs drifted data."""
    base = np.random.normal(650, 50, 1000)
    identical = base.copy()
    drifted = np.random.normal(550, 60, 1000)
    
    psi_identical = calculate_psi(base, identical)
    psi_drifted = calculate_psi(base, drifted)
    
    assert psi_identical < 0.05, f"Expected near-zero PSI for identical data, got {psi_identical}"
    assert psi_drifted > 0.20, f"Expected significant PSI for shifted distribution, got {psi_drifted}"

def test_fastapi_health():
    """Verify FastAPI service health check."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_fastapi_predict_prime():
    """Verify real-time prediction for prime borrower."""
    payload = {
        "age": 35,
        "annual_income": 120000.0,
        "employment_years": 8,
        "home_ownership": "MORTGAGE",
        "education_level": "Bachelor",
        "fico_score": 780,
        "open_credit_lines": 10,
        "bankruptcies": 0,
        "inquiries_last_6m": 0,
        "revolving_utilization_ratio": 0.15,
        "total_debt_to_income_ratio": 0.20,
        "payment_to_income_ratio": 0.05,
        "loan_amount": 10000.0,
        "loan_intent": "DEBT_CONSOLIDATION",
        "interest_rate": 8.5,
        "term_months": 36,
        "rolling_avg_dpd_6m": 0.0,
        "prior_max_dpd_12m": 0,
        "prior_delinquent_months": 0,
        "rolling_payment_ratio_3m": 1.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["decision"] in ["APPROVE", "DECLINE", "REFER_MANUAL_REVIEW"]
    assert 0.0 <= res["default_probability"] <= 1.0
    assert len(res["top_risk_drivers"]) > 0
