"""
CreditPulse-AI Interactive Executive & Underwriting Cockpit
Streamlit application featuring:
1. Real-Time Underwriter Studio & Adverse Action SHAP Explanations
2. DuckDB Live SQL Analytics Console
3. Model Diagnostics & Cost-Sensitive Profit Economics
4. MLOps Drift & Governance Center (PSI)
"""

import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.query_runner import (
    execute_custom_query, get_cohort_risk_analysis,
    get_loan_intent_profitability
)
from src.ml.explainability import explain_single_applicant
from src.ml.drift import evaluate_portfolio_drift, generate_macro_shock_batch
import joblib

MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(
    page_title="CreditPulse-AI | Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2b5c8f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .status-approve {
        background-color: #d4edda;
        color: #155724;
        padding: 12px 20px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
    }
    .status-decline {
        background-color: #f8d7da;
        color: #721c24;
        padding: 12px 20px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
    }
    .status-review {
        background-color: #fff3cd;
        color: #856404;
        padding: 12px 20px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ml_assets():
    """Loads model, metadata, and threshold configuration."""
    try:
        model = joblib.load(MODELS_DIR / "calibrated_model.joblib")
        with open(MODELS_DIR / "feature_metadata.json", "r") as f:
            metadata = json.load(f)
        with open(MODELS_DIR / "threshold_optimization.json", "r") as f:
            thresh_config = json.load(f)
        with open(MODELS_DIR / "benchmark_results.json", "r") as f:
            benchmark = json.load(f)
        return model, metadata, thresh_config, benchmark
    except Exception as e:
        return None, None, None, None

model, metadata, thresh_config, benchmark = load_ml_assets()

# Header
st.title("🏦 CreditPulse-AI: Risk & Decision Intelligence")
st.caption("Production Underwriting Engine • DuckDB SQL Warehouse • Cost-Sensitive Optimization • Regulatory SHAP • PSI Drift Monitoring")

if model is None:
    st.error("⚠️ Model artifacts not found. Please run `python run_pipeline.py` first to generate models and warehouse tables.")
    st.stop()

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Underwriting Studio",
    "⚡ DuckDB SQL Console",
    "📈 Decision Economics & Models",
    "🛡️ MLOps & Drift Monitor"
])

# -----------------------------------------------------------------------------
# TAB 1: UNDERWRITING STUDIO
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Interactive Loan Underwriting & Adverse Action Explainer")
    st.write("Score loan applicants in real-time using calibrated ensemble ML and cost-sensitive policy thresholds.")

    # Preset Profiles
    col_preset1, col_preset2, col_preset3 = st.columns([1, 1, 1])
    preset = st.radio(
        "Load Sample Applicant Profile:",
        ["Custom Input", "Prime Applicant (Low Risk)", "Borderline Applicant (Near-Prime)", "Stressed Applicant (Subprime)"],
        horizontal=True
    )

    # Default values based on preset
    if preset == "Prime Applicant (Low Risk)":
        d_fico, d_inc, d_amt, d_util, d_dti, d_dpd, d_emp = 765, 115000, 15000, 0.12, 0.22, 0.0, 9
    elif preset == "Borderline Applicant (Near-Prime)":
        d_fico, d_inc, d_amt, d_util, d_dti, d_dpd, d_emp = 640, 58000, 18000, 0.48, 0.41, 5.0, 3
    elif preset == "Stressed Applicant (Subprime)":
        d_fico, d_inc, d_amt, d_util, d_dti, d_dpd, d_emp = 530, 36000, 22000, 0.88, 0.65, 25.0, 1
    else:
        d_fico, d_inc, d_amt, d_util, d_dti, d_dpd, d_emp = 680, 75000, 16000, 0.35, 0.34, 0.0, 5

    with st.form("underwrite_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 👤 Borrower Demographics")
            age = st.slider("Age", 21, 70, 34)
            income = st.number_input("Annual Income ($)", min_value=15000, max_value=400000, value=d_inc, step=5000)
            emp_years = st.slider("Years Employed", 0, 35, d_emp)
            home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
            education = st.selectbox("Education Level", ["Bachelor", "High School", "Master", "PhD"])

        with c2:
            st.markdown("##### 💳 Credit Bureau Profile")
            fico = st.slider("FICO Score", 350, 850, d_fico)
            open_lines = st.slider("Open Credit Lines", 1, 25, 8)
            utilization = st.slider("Revolving Utilization Ratio", 0.0, 1.2, float(d_util), step=0.05)
            bankruptcies = st.selectbox("Prior Bankruptcies", [0, 1, 2], index=0)
            inquiries = st.slider("Inquiries (Last 6 Months)", 0, 8, 1)

        with c3:
            st.markdown("##### 📝 Loan Terms & Repayment")
            loan_amt = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=50000, value=d_amt, step=1000)
            intent = st.selectbox("Loan Purpose", ["DEBT_CONSOLIDATION", "HOME_IMPROVEMENT", "VENTURE", "MEDICAL", "PERSONAL"])
            rate = st.slider("Interest Rate (%)", 5.0, 28.0, 13.5, step=0.25)
            term = st.selectbox("Term Length", [36, 60], index=0)
            rolling_dpd = st.number_input("Rolling 6-Month Avg DPD", min_value=0.0, max_value=90.0, value=float(d_dpd), step=1.0)
            dti = st.slider("Total Debt-to-Income (DTI)", 0.05, 1.2, float(d_dti), step=0.02)

        submit = st.form_submit_button("⚡ Run Real-Time Underwriting Assessment", use_container_width=True)

    # Derived Ratios
    pti = round((loan_amt / term) / (income / 12.0), 4)

    applicant_data = {
        "age": age,
        "annual_income": income,
        "employment_years": emp_years,
        "home_ownership": home_ownership,
        "education_level": education,
        "fico_score": fico,
        "open_credit_lines": open_lines,
        "bankruptcies": bankruptcies,
        "inquiries_last_6m": inquiries,
        "revolving_utilization_ratio": utilization,
        "total_debt_to_income_ratio": dti,
        "payment_to_income_ratio": pti,
        "loan_amount": loan_amt,
        "loan_intent": intent,
        "interest_rate": rate,
        "term_months": term,
        "rolling_avg_dpd_6m": rolling_dpd,
        "prior_max_dpd_12m": int(rolling_dpd * 1.5),
        "prior_delinquent_months": 1 if rolling_dpd > 10 else 0,
        "rolling_payment_ratio_3m": 0.85 if rolling_dpd > 15 else 1.0
    }

    # Run Prediction
    df_eval = pd.DataFrame([applicant_data])
    prob_default = float(model.predict_proba(df_eval)[0, 1])
    optimal_tau = thresh_config["optimal_threshold"]

    # Decisioning
    if prob_default < optimal_tau:
        decision_label = "APPROVED"
        decision_class = "status-approve"
        decision_desc = f"Applicant default risk ({prob_default:.1%}) is safely below policy threshold ({optimal_tau:.1%})."
    elif prob_default < (optimal_tau + 0.06):
        decision_label = "REFER TO MANUAL UNDERWRITING"
        decision_class = "status-review"
        decision_desc = f"Borderline application ({prob_default:.1%}) within review band [{optimal_tau:.1%} - {optimal_tau+0.06:.1%}]."
    else:
        decision_label = "DECLINED"
        decision_class = "status-decline"
        decision_desc = f"Default probability ({prob_default:.1%}) exceeds policy cutoff ({optimal_tau:.1%})."

    # Financial Expectation
    exp_interest = loan_amt * (rate / 100.0) * (term / 12.0) * 0.80
    exp_loss = loan_amt * 0.70
    exp_profit = (1.0 - prob_default) * exp_interest - prob_default * exp_loss

    st.markdown("---")
    st.markdown(f'<div class="{decision_class}">{decision_label}</div>', unsafe_allow_html=True)
    st.caption(decision_desc)

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Default Probability (PD)", f"{prob_default:.2%}", delta=f"{prob_default - optimal_tau:+.1%} vs Policy Cutoff", delta_color="inverse")
    m2.metric("Optimal Policy Cutoff (tau*)", f"{optimal_tau:.1%}")
    m3.metric("Expected Net Profit", f"${exp_profit:,.2f}", delta="Risk-Adjusted Return")
    m4.metric("Payment-to-Income (PTI)", f"{pti:.1%}")

    # Adverse Action SHAP Factors
    st.markdown("##### ⚖️ Equal Credit Opportunity Act (ECOA) Adverse Action Attribution")
    st.caption("Explains exact mathematical factors driving this borrower's risk score (SHAP feature contributions).")
    
    with st.spinner("Generating SHAP feature waterfall..."):
        reasons = explain_single_applicant(applicant_data)
    
    df_reasons = pd.DataFrame(reasons[:8])
    chart = alt.Chart(df_reasons).mark_bar().encode(
        x=alt.X("shap_value:Q", title="SHAP Contribution to Default Risk"),
        y=alt.Y("feature:N", sort="-x", title="Feature"),
        color=alt.Color("impact:N", scale=alt.Scale(domain=["Increases Risk", "Decreases Risk"], range=["#e63946", "#2a9d8f"])),
        tooltip=["feature", "shap_value", "impact"]
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: DUCKDB SQL CONSOLE
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("⚡ DuckDB Embedded Data Warehouse & Feature Store")
    st.write("Query 10,000 borrowers, 120,000 payment ledger records, and computed rolling window features with lightning speed.")

    q_col1, q_col2, q_col3 = st.columns(3)
    prefill = None
    if q_col1.button("📊 FICO Tier & DTI Cohort Risk"):
        prefill = """SELECT 
    CASE 
        WHEN fico_score < 580 THEN '1. Poor (<580)'
        WHEN fico_score BETWEEN 580 AND 669 THEN '2. Fair (580-669)'
        WHEN fico_score BETWEEN 670 AND 739 THEN '3. Good (670-739)'
        ELSE '4. Excellent (740+)'
    END AS fico_tier,
    COUNT(*) AS borrowers,
    ROUND(AVG(is_default) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(revolving_utilization_ratio) * 100.0, 1) AS avg_utilization_pct,
    ROUND(AVG(annual_income), 0) AS avg_income
FROM v_credit_features
GROUP BY 1
ORDER BY 1;"""

    if q_col2.button("💼 Loan Purpose Profitability"):
        prefill = """SELECT 
    loan_intent,
    COUNT(*) AS application_volume,
    ROUND(AVG(interest_rate), 2) AS avg_rate_pct,
    ROUND(AVG(loan_amount), 0) AS avg_amount,
    ROUND(AVG(is_default) * 100.0, 2) AS default_rate_pct,
    ROUND(SUM(loan_amount), 0) AS total_funded_usd
FROM v_credit_features
GROUP BY 1
ORDER BY default_rate_pct DESC;"""

    if q_col3.button("⏱️ Rolling Delinquency Trajectory (Window)"):
        prefill = """SELECT 
    customer_id,
    month_seq,
    days_past_due,
    AVG(days_past_due) OVER (
        PARTITION BY customer_id 
        ORDER BY month_seq 
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) AS rolling_6m_avg_dpd,
    payment_status
FROM payment_ledger
WHERE days_past_due > 0
LIMIT 20;"""

    query_input = st.text_area(
        "SQL Query Editor:",
        value=prefill if prefill else "SELECT customer_id, age, annual_income, fico_score, revolving_utilization_ratio, is_default FROM v_credit_features LIMIT 25;",
        height=140
    )

    if st.button("▶ Run SQL Query", type="primary"):
        try:
            df_sql = execute_custom_query(query_input)
            st.success(f"Query returned {len(df_sql)} rows.")
            st.dataframe(df_sql, use_container_width=True)
        except Exception as e:
            st.error(f"SQL Error: {e}")

# -----------------------------------------------------------------------------
# TAB 3: DECISION ECONOMICS & MODELS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("📈 Decision Economics & Multi-Model Benchmark")
    st.write("Demonstrating why naive 0.5 classification cutoffs leave money on the table.")

    col_bench1, col_bench2 = st.columns([1, 1])
    with col_bench1:
        st.markdown("##### 🏆 Model Benchmark Comparison")
        records = []
        for m_name, m_data in benchmark.items():
            records.append({
                "Model": m_name.replace("_", " "),
                "Test ROC-AUC": m_data["test"]["roc_auc"],
                "Test PR-AUC": m_data["test"]["pr_auc"],
                "Test F1-Score": m_data["test"]["f1_score"],
                "Brier Score (Lower=Better)": m_data["test"]["brier_score"]
            })
        st.dataframe(pd.DataFrame(records).set_index("Model"), use_container_width=True)

    with col_bench2:
        st.markdown("##### 💰 Cost-Sensitive Optimization Summary")
        st.write(f"• **Optimal Threshold (tau*)**: `{thresh_config['optimal_threshold']:.2f}`")
        st.write(f"• **Net Profit at tau***: `${thresh_config['optimal_profit']:,.2f}` (Approval Rate: {thresh_config['optimal_approval_rate']:.1%})")
        st.write(f"• **Net Profit at Naive 0.50**: `${thresh_config['naive_profit']:,.2f}` (Approval Rate: {thresh_config['naive_approval_rate']:.1%})")
        st.success(f"🚀 **Incremental Profit Added**: +${thresh_config['incremental_profit_dollars']:,.2f} (+{thresh_config['percentage_profit_improvement']}%)")

    st.markdown("---")
    st.markdown("##### 📊 Portfolio Profit Curve Across Thresholds (tau)")
    df_curve = pd.DataFrame(thresh_config["simulation_curve"])
    
    profit_chart = alt.Chart(df_curve).mark_line(color="#2b5c8f", strokeWidth=3).encode(
        x=alt.X("threshold:Q", title="Decision Cutoff (tau)"),
        y=alt.Y("net_profit:Q", title="Expected Net Portfolio Profit ($)"),
        tooltip=["threshold", "net_profit", "approval_rate", "approved_default_rate"]
    ).properties(height=300)
    st.altair_chart(profit_chart, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: MLOPS DRIFT & GOVERNANCE
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🛡️ MLOps Drift & Governance Center")
    st.write("Real-time monitoring of covariate shift using the banking standard **Population Stability Index (PSI)**.")

    st.info("""
    **Banking Industry Standard (OCC/Fed Regulatory Thresholds):**
    - **PSI < 0.10**: Stable (Distribution unchanged)
    - **0.10 ≤ PSI < 0.25**: Moderate Shift (Warning - Monitor cohort)
    - **PSI ≥ 0.25**: Significant Drift (Critical - Automatic trigger for model recalibration/retraining)
    """)

    st.markdown("##### ⚡ Macroeconomic Shock Simulator")
    st.caption("Simulate an economic recession / inflation surge to test if the monitoring engine triggers alarms.")
    
    shock = st.slider("Inject Macroeconomic Shock Severity:", min_value=0.0, max_value=1.0, value=0.0, step=0.1, format="%d%%")
    
    test_path = MODELS_DIR / "holdout_test.parquet"
    if test_path.exists():
        df_base = pd.read_parquet(test_path)
        if shock > 0.0:
            df_current = generate_macro_shock_batch(df_base, shock_severity=shock)
        else:
            df_current = df_base.sample(frac=0.5, random_state=42)

        monitored_features = [
            "fico_score", "revolving_utilization_ratio", "total_debt_to_income_ratio",
            "annual_income", "rolling_avg_dpd_6m", "open_credit_lines"
        ]
        drift_report = evaluate_portfolio_drift(df_base, df_current, monitored_features)

        # Status badge
        status = drift_report["overall_portfolio_status"]
        if status == "HEALTHY":
            st.success("🟢 **PORTFOLIO HEALTH: HEALTHY (All features PSI < 0.10)**")
        elif status == "MONITOR_CLOSELY":
            st.warning("🟡 **PORTFOLIO HEALTH: MONITOR CLOSELY (Moderate covariate drift detected)**")
        else:
            st.error("🔴 **PORTFOLIO HEALTH: ACTION REQUIRED (Critical drift PSI ≥ 0.25 - Recalibration required)**")

        st.dataframe(pd.DataFrame(drift_report["feature_metrics"]), use_container_width=True)
