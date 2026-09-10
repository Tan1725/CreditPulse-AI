-- ============================================================================
-- CreditPulse-AI: Advanced SQL Feature Engineering Pipeline
-- Computes behavioral, bureau, and application features using Window Functions,
-- CTEs, and financial ratio transformations.
-- ============================================================================

CREATE OR REPLACE VIEW v_credit_features AS
WITH ledger_aggregations AS (
    -- 1. Compute rolling window metrics and temporal delinquency trajectories
    SELECT 
        customer_id,
        application_id,
        month_seq,
        days_past_due,
        amount_paid,
        amount_due,
        -- Rolling 6-month average days past due (captures recent payment distress)
        AVG(days_past_due) OVER (
            PARTITION BY customer_id 
            ORDER BY month_seq 
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) AS rolling_avg_dpd_6m,
        -- Maximum delinquency severity observed
        MAX(days_past_due) OVER (
            PARTITION BY customer_id
        ) AS max_dpd_12m,
        -- Count of severe delinquencies (30+ days past due)
        SUM(CASE WHEN days_past_due >= 30 THEN 1 ELSE 0 END) OVER (
            PARTITION BY customer_id
        ) AS count_delinquent_months,
        -- Payment completeness ratio across recent 3 months
        AVG(amount_paid / NULLIF(amount_due, 0)) OVER (
            PARTITION BY customer_id 
            ORDER BY month_seq 
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_payment_ratio_3m
    FROM payment_ledger
),

latest_ledger_snapshot AS (
    -- 2. Extract final state per borrower (snapshot at month 12)
    SELECT 
        customer_id,
        application_id,
        rolling_avg_dpd_6m,
        max_dpd_12m,
        count_delinquent_months,
        rolling_payment_ratio_3m,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY month_seq DESC) as rn
    FROM ledger_aggregations
),

customer_bureau_profile AS (
    -- 3. Join customer demographics, bureau telemetry, and loan applications
    SELECT 
        c.customer_id,
        c.age,
        c.annual_income,
        c.employment_years,
        c.home_ownership,
        c.education_level,
        b.fico_score,
        b.open_credit_lines,
        b.total_credit_limit,
        b.revolving_balance,
        b.bankruptcies,
        b.inquiries_last_6m,
        a.application_id,
        a.loan_amount,
        a.loan_intent,
        a.interest_rate,
        a.term_months,
        a.loan_status,
        a.is_default,
        l.rolling_avg_dpd_6m,
        l.max_dpd_12m AS prior_max_dpd_12m,
        l.count_delinquent_months AS prior_delinquent_months,
        l.rolling_payment_ratio_3m
    FROM customers c
    INNER JOIN credit_bureau b ON c.customer_id = b.customer_id
    INNER JOIN loan_applications a ON c.customer_id = a.customer_id
    INNER JOIN latest_ledger_snapshot l ON c.customer_id = l.customer_id AND l.rn = 1
)

-- 4. Final Feature Set with Non-linear Financial Ratios & Target Flag
SELECT 
    customer_id,
    application_id,
    
    -- Demographics & Stability
    age,
    annual_income,
    employment_years,
    home_ownership,
    education_level,
    
    -- Credit Bureau Profile
    fico_score,
    open_credit_lines,
    bankruptcies,
    inquiries_last_6m,
    
    -- Financial Stress & Capacity Ratios
    ROUND(revolving_balance / NULLIF(total_credit_limit, 0), 4) AS revolving_utilization_ratio,
    ROUND((loan_amount + revolving_balance) / NULLIF(annual_income, 0), 4) AS total_debt_to_income_ratio,
    ROUND((loan_amount / NULLIF(term_months, 0)) / NULLIF(annual_income / 12.0, 0), 4) AS payment_to_income_ratio,
    
    -- Loan Characteristics
    loan_amount,
    loan_intent,
    interest_rate,
    term_months,
    loan_status,
    
    -- Historical Repayment Behavior on Prior Trade Lines (from SQL Window Functions)
    ROUND(rolling_avg_dpd_6m, 2) AS rolling_avg_dpd_6m,
    prior_max_dpd_12m,
    prior_delinquent_months,
    ROUND(rolling_payment_ratio_3m, 4) AS rolling_payment_ratio_3m,
    
    -- TARGET VARIABLE: 1 if borrower defaulted on this loan, 0 otherwise
    is_default

FROM customer_bureau_profile;
