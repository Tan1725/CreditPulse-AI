"""
CreditPulse-AI API Request and Response Schemas
Type-safe data validation contracts using Pydantic.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class ApplicantFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    annual_income: float = Field(..., ge=1000, description="Gross annual income in USD")
    employment_years: int = Field(..., ge=0, le=50, description="Years in current employment")
    home_ownership: str = Field(default="RENT", description="RENT, MORTGAGE, OWN, OTHER")
    education_level: str = Field(default="Bachelor", description="High School, Bachelor, Master, PhD")
    
    fico_score: int = Field(..., ge=300, le=850, description="Credit bureau FICO score")
    open_credit_lines: int = Field(..., ge=0, le=50, description="Active trade lines")
    bankruptcies: int = Field(default=0, ge=0, le=10, description="Past bankruptcies recorded")
    inquiries_last_6m: int = Field(default=0, ge=0, le=20, description="Hard credit inquiries in last 6 months")
    
    revolving_utilization_ratio: float = Field(..., ge=0.0, le=2.0, description="Revolving balance / credit limit")
    total_debt_to_income_ratio: float = Field(..., ge=0.0, le=3.0, description="Total debt / annual income")
    payment_to_income_ratio: float = Field(..., ge=0.0, le=1.5, description="Monthly loan installment / monthly income")
    
    loan_amount: float = Field(..., ge=500, description="Requested principal amount in USD")
    loan_intent: str = Field(default="DEBT_CONSOLIDATION", description="Loan intent / purpose")
    interest_rate: float = Field(..., ge=1.0, le=40.0, description="Offered annual interest rate in percent")
    term_months: int = Field(default=36, description="Loan repayment duration in months")
    
    rolling_avg_dpd_6m: float = Field(default=0.0, ge=0.0, description="Rolling 6-month average days past due")
    prior_max_dpd_12m: int = Field(default=0, ge=0, description="Maximum days past due in prior 12 months")
    prior_delinquent_months: int = Field(default=0, ge=0, description="Count of months with >=30 DPD")
    rolling_payment_ratio_3m: float = Field(default=1.0, ge=0.0, le=2.0, description="Payment completeness ratio")

class RiskFactor(BaseModel):
    feature: str
    shap_value: float
    impact: str

class UnderwritingDecisionResponse(BaseModel):
    decision: str = Field(..., description="APPROVE, DECLINE, or MANUAL_REVIEW")
    default_probability: float = Field(..., description="Calibrated empirical probability of default")
    optimal_threshold: float = Field(..., description="Policy decision threshold (tau*)")
    risk_tier: str = Field(..., description="Risk categorization: Prime, Near-Prime, Subprime")
    expected_net_profit: float = Field(..., description="Expected monetary return if funded")
    top_risk_drivers: List[RiskFactor] = Field(..., description="Adverse action regulatory factors")
    inference_latency_ms: float

class SqlQueryRequest(BaseModel):
    query: str = Field(..., description="Read-only SQL query to execute against DuckDB warehouse")

class SqlQueryResponse(BaseModel):
    columns: List[str]
    row_count: int
    data: List[dict]
