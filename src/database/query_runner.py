"""
CreditPulse-AI Query Runner & Analytical Engine
Executes feature engineering scripts, provides dataset loaders,
and runs pre-packaged executive risk queries against DuckDB.
"""

from pathlib import Path
import pandas as pd
import duckdb
from src.database.db_manager import get_connection, DB_PATH

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"

def build_features_view(conn: duckdb.DuckDBPyConnection = None) -> None:
    """Reads and executes feature_engineering.sql to build the feature store view."""
    close_conn = False
    if conn is None:
        conn = get_connection(read_only=False)
        close_conn = True
    
    sql_file = SQL_DIR / "feature_engineering.sql"
    with open(sql_file, "r") as f:
        sql_script = f.read()
    
    conn.execute(sql_script)
    if close_conn:
        conn.close()

def load_modeling_dataset() -> pd.DataFrame:
    """Loads the feature store view into a pandas DataFrame."""
    build_features_view()
    conn = get_connection(read_only=True)
    query = "SELECT * FROM v_credit_features"
    df = conn.execute(query).df()
    conn.close()
    return df

def execute_custom_query(sql_query: str) -> pd.DataFrame:
    """Executes arbitrary read-only SQL queries on the warehouse."""
    build_features_view()
    conn = get_connection(read_only=True)
    df = conn.execute(sql_query).df()
    conn.close()
    return df

def get_cohort_risk_analysis() -> pd.DataFrame:
    """Pre-built SQL query: Default rates binned across FICO tiers and DTI categories."""
    query = """
    SELECT 
        CASE 
            WHEN fico_score < 580 THEN '1. Poor (<580)'
            WHEN fico_score BETWEEN 580 AND 669 THEN '2. Fair (580-669)'
            WHEN fico_score BETWEEN 670 AND 739 THEN '3. Good (670-739)'
            ELSE '4. Excellent (740+)'
        END AS fico_tier,
        CASE 
            WHEN total_debt_to_income_ratio > 0.50 THEN 'High DTI (>50%)'
            WHEN total_debt_to_income_ratio BETWEEN 0.35 AND 0.50 THEN 'Moderate DTI (35-50%)'
            ELSE 'Healthy DTI (<35%)'
        END AS dti_category,
        COUNT(*) AS total_applicants,
        SUM(is_default) AS total_defaults,
        ROUND(AVG(is_default) * 100.0, 2) AS default_rate_pct,
        ROUND(AVG(loan_amount), 0) AS avg_loan_amount,
        ROUND(AVG(revolving_utilization_ratio) * 100.0, 1) AS avg_utilization_pct
    FROM v_credit_features
    GROUP BY 1, 2
    ORDER BY 1, 2;
    """
    return execute_custom_query(query)

def get_loan_intent_profitability() -> pd.DataFrame:
    """Pre-built SQL query: Analysis by loan intent."""
    query = """
    SELECT 
        loan_intent,
        COUNT(*) as applicant_count,
        ROUND(AVG(interest_rate), 2) as avg_interest_rate,
        ROUND(AVG(loan_amount), 0) as avg_loan_amount,
        ROUND(AVG(is_default) * 100.0, 2) as default_rate_pct,
        ROUND(SUM(loan_amount), 0) as total_funded_volume
    FROM v_credit_features
    GROUP BY 1
    ORDER BY default_rate_pct DESC;
    """
    return execute_custom_query(query)

if __name__ == "__main__":
    df = load_modeling_dataset()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Default rate: {df['is_default'].mean():.2%}")
