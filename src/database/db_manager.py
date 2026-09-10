"""
CreditPulse-AI Database Manager
Manages DuckDB initialization, multi-table relational schema creation,
and generation of high-fidelity financial transaction and credit bureau data.
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DATA_DIR / "credit_warehouse.duckdb"

def get_connection(db_path: Path = DB_PATH, read_only: bool = False):
    """Obtain a connection to the DuckDB warehouse."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path), read_only=read_only)

def init_schema(conn: duckdb.DuckDBPyConnection):
    """Create normalized relational tables."""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id VARCHAR PRIMARY KEY,
        age INTEGER,
        annual_income DOUBLE,
        employment_years INTEGER,
        home_ownership VARCHAR,
        state VARCHAR,
        education_level VARCHAR
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS credit_bureau (
        bureau_id VARCHAR PRIMARY KEY,
        customer_id VARCHAR,
        fico_score INTEGER,
        open_credit_lines INTEGER,
        total_credit_limit DOUBLE,
        revolving_balance DOUBLE,
        bankruptcies INTEGER,
        inquiries_last_6m INTEGER,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS loan_applications (
        application_id VARCHAR PRIMARY KEY,
        customer_id VARCHAR,
        loan_amount DOUBLE,
        loan_intent VARCHAR,
        interest_rate DOUBLE,
        term_months INTEGER,
        application_date DATE,
        loan_status VARCHAR,
        is_default INTEGER,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS payment_ledger (
        ledger_id VARCHAR PRIMARY KEY,
        customer_id VARCHAR,
        application_id VARCHAR,
        month_seq INTEGER,
        amount_due DOUBLE,
        amount_paid DOUBLE,
        days_past_due INTEGER,
        payment_status VARCHAR,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
        FOREIGN KEY (application_id) REFERENCES loan_applications(application_id)
    );
    """)

def generate_synthetic_data(n_customers: int = 10000, seed: int = 42):
    """
    Generates realistic, mathematically correlated financial and repayment data.
    Simulates real-world credit risk dynamics:
    - High utilization & low FICO correlate with higher probability of default.
    - Monthly payment ledger tracks delinquency history (DPD) over 12 months.
    """
    np.random.seed(seed)
    print(f"Generating realistic credit portfolio for {n_customers:,} borrowers...")

    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
    
    # 1. Customers Table
    ages = np.random.randint(21, 68, size=n_customers)
    # Lognormal income distribution
    incomes = np.round(np.random.lognormal(mean=10.9, sigma=0.55, size=n_customers), -2)
    incomes = np.clip(incomes, 18000, 350000)
    
    emp_years = np.clip((ages - 18) * np.random.beta(2, 5, size=n_customers), 0, 40).astype(int)
    home_ownership = np.random.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], size=n_customers, p=[0.48, 0.40, 0.10, 0.02])
    states = np.random.choice(["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "WA"], size=n_customers)
    education = np.random.choice(["High School", "Bachelor", "Master", "PhD"], size=n_customers, p=[0.35, 0.45, 0.16, 0.04])

    df_customers = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "annual_income": incomes,
        "employment_years": emp_years,
        "home_ownership": home_ownership,
        "state": states,
        "education_level": education
    })

    # 2. Credit Bureau Table
    # FICO score conditioned partially on age and income
    base_fico = np.random.normal(loc=675, scale=65, size=n_customers)
    income_bonus = np.clip((incomes - 50000) / 10000 * 4, -40, 50)
    fico_scores = np.clip(np.round(base_fico + income_bonus), 320, 850).astype(int)

    open_lines = np.clip(np.random.poisson(lam=7, size=n_customers) + (fico_scores - 600) // 40, 1, 30).astype(int)
    total_limits = np.round(np.clip(incomes * np.random.uniform(0.15, 0.65, size=n_customers), 2000, 120000), -2)
    
    # Utilization is strongly inverse to FICO score
    utilization_rate = np.clip(np.random.beta(2, 5, size=n_customers) + (800 - fico_scores) / 700, 0.02, 0.98)
    revolving_balances = np.round(total_limits * utilization_rate, 2)

    # Bankruptcies and inquiries correlated with lower FICO
    bankruptcies = np.where(fico_scores < 580, np.random.choice([0, 1, 2], size=n_customers, p=[0.75, 0.20, 0.05]), 0)
    inquiries = np.clip(np.random.poisson(lam=1.5, size=n_customers) + np.where(fico_scores < 620, 2, 0), 0, 12).astype(int)

    df_bureau = pd.DataFrame({
        "bureau_id": [f"BUR_{i:06d}" for i in range(1, n_customers + 1)],
        "customer_id": customer_ids,
        "fico_score": fico_scores,
        "open_credit_lines": open_lines,
        "total_credit_limit": total_limits,
        "revolving_balance": revolving_balances,
        "bankruptcies": bankruptcies,
        "inquiries_last_6m": inquiries
    })

    # 3. Loan Applications Table
    app_ids = [f"APP_{i:06d}" for i in range(1, n_customers + 1)]
    loan_amounts = np.round(np.clip(incomes * np.random.uniform(0.1, 0.4, size=n_customers), 1000, 45000), -2)
    loan_intents = np.random.choice(
        ["DEBT_CONSOLIDATION", "HOME_IMPROVEMENT", "VENTURE", "MEDICAL", "EDUCATION", "PERSONAL"],
        size=n_customers,
        p=[0.45, 0.18, 0.12, 0.10, 0.08, 0.07]
    )
    # Interest rate depends on FICO score
    base_rate = 22.0 - (fico_scores - 350) * (16.0 / 500.0)
    interest_rates = np.clip(np.round(base_rate + np.random.normal(0, 1.2, size=n_customers), 2), 5.5, 26.5)
    terms = np.random.choice([36, 60], size=n_customers, p=[0.70, 0.30])
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=int(d)) for d in np.random.randint(0, 365, size=n_customers)]

    # Latent default risk probability based on economic indicators
    dti_proxy = (revolving_balances + loan_amounts) / incomes
    utilization_proxy = revolving_balances / np.maximum(total_limits, 1.0)
    risk_logits = (
        - 2.2
        - 0.015 * (fico_scores - 660)
        + 2.6 * (dti_proxy - 0.35)
        + 2.2 * (utilization_proxy - 0.40)
        + 0.22 * inquiries
        + 0.65 * bankruptcies
        - 0.025 * emp_years
    )
    default_probs = 1.0 / (1.0 + np.exp(-risk_logits))
    
    # Stochastic loan performance outcome (realistic non-deterministic ground truth)
    is_default = (np.random.rand(n_customers) < default_probs).astype(int)
    loan_status = np.where(is_default == 1, "CHARGEOFF_DEFAULT", "PAID_OFF")

    df_apps = pd.DataFrame({
        "application_id": app_ids,
        "customer_id": customer_ids,
        "loan_amount": loan_amounts,
        "loan_intent": loan_intents,
        "interest_rate": interest_rates,
        "term_months": terms,
        "application_date": dates,
        "loan_status": loan_status,
        "is_default": is_default
    })

    # 4. Payment Ledger (12 months of pre-existing credit account repayment behavior)
    print("Simulating 12-month historical trade line repayment ledger (120,000 transactions)...")
    ledger_records = []
    
    ledger_counter = 1
    for i in range(n_customers):
        c_id = customer_ids[i]
        a_id = app_ids[i]
        p_def = default_probs[i]
        amt = loan_amounts[i]
        term = terms[i]
        monthly_payment = round(amt / term * 1.15, 2)
        
        has_defaulted = False
        current_dpd = 0
        
        for m in range(1, 13):
            if has_defaulted:
                # Once defaulted, payments cease
                current_dpd += 30
                status = "DEFAULT"
                paid = 0.0
            else:
                # Probability of being late in month m
                p_late = min(0.85, p_def * 0.65)
                if np.random.rand() < p_late:
                    added_dpd = np.random.choice([15, 30, 45, 60], p=[0.55, 0.25, 0.12, 0.08])
                    current_dpd = min(180, current_dpd + added_dpd)
                    if current_dpd >= 90:
                        has_defaulted = True
                        status = "DEFAULT"
                        paid = 0.0
                    elif current_dpd >= 60:
                        status = "LATE_60"
                        paid = round(monthly_payment * 0.5, 2)
                    elif current_dpd >= 30:
                        status = "LATE_30"
                        paid = round(monthly_payment * 0.75, 2)
                    else:
                        status = "LATE_15"
                        paid = monthly_payment
                else:
                    current_dpd = max(0, current_dpd - 30)
                    status = "ON_TIME"
                    paid = monthly_payment

            ledger_records.append((
                f"TXN_{ledger_counter:08d}",
                c_id,
                a_id,
                m,
                monthly_payment,
                paid,
                current_dpd,
                status
            ))
            ledger_counter += 1

    df_ledger = pd.DataFrame(ledger_records, columns=[
        "ledger_id", "customer_id", "application_id", "month_seq",
        "amount_due", "amount_paid", "days_past_due", "payment_status"
    ])

    return df_customers, df_bureau, df_apps, df_ledger

def populate_database(n_customers: int = 10000, force_recreate: bool = True):
    """Seed DuckDB warehouse with generated tables."""
    conn = get_connection()
    if force_recreate:
        conn.execute("DROP TABLE IF EXISTS payment_ledger;")
        conn.execute("DROP TABLE IF EXISTS loan_applications;")
        conn.execute("DROP TABLE IF EXISTS credit_bureau;")
        conn.execute("DROP TABLE IF EXISTS customers;")
    
    init_schema(conn)
    
    df_customers, df_bureau, df_apps, df_ledger = generate_synthetic_data(n_customers)
    
    print("Writing records into DuckDB...")
    conn.register("df_customers_view", df_customers)
    conn.execute("INSERT INTO customers SELECT * FROM df_customers_view")
    
    conn.register("df_bureau_view", df_bureau)
    conn.execute("INSERT INTO credit_bureau SELECT * FROM df_bureau_view")
    
    conn.register("df_apps_view", df_apps)
    conn.execute("INSERT INTO loan_applications SELECT * FROM df_apps_view")
    
    conn.register("df_ledger_view", df_ledger)
    conn.execute("INSERT INTO payment_ledger SELECT * FROM df_ledger_view")
    
    print(f"Successfully loaded {conn.execute('SELECT COUNT(*) FROM customers').fetchone()[0]:,} customers.")
    print(f"Successfully loaded {conn.execute('SELECT COUNT(*) FROM payment_ledger').fetchone()[0]:,} ledger records.")
    conn.close()

if __name__ == "__main__":
    populate_database(n_customers=10000)
