"""
CreditPulse-AI Statistical Drift & Governance Engine (PSI & KS-Test)
Monitors feature-level covariate drift and prediction score drift using
the regulatory banking standard: Population Stability Index (PSI) and Kolmogorov-Smirnov.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10, epsilon: float = 1e-4) -> float:
    """
    Computes Population Stability Index (PSI) between baseline and production distributions.
    
    Rule of thumb:
      PSI < 0.10: Stable (No Action)
      0.10 <= PSI < 0.25: Moderate Shift (Warning)
      PSI >= 0.25: Significant Shift (Alert / Retrain Trigger)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine bin percentiles based on baseline (expected)
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges = np.unique(bin_edges)
    
    if len(bin_edges) < 2:
        return 0.0
    
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate bucket counts
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to proportions
    expected_pct = (expected_counts / len(expected)) + epsilon
    actual_pct = (actual_counts / len(actual)) + epsilon

    # Normalize after epsilon smoothing
    expected_pct /= np.sum(expected_pct)
    actual_pct /= np.sum(actual_pct)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.round(psi_val, 4))

def evaluate_portfolio_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame, feature_list: list) -> dict:
    """Evaluates PSI and KS-test statistics across all numerical features."""
    drift_records = []
    
    for feat in feature_list:
        if feat not in baseline_df.columns or feat not in current_df.columns:
            continue
            
        base_vals = baseline_df[feat].dropna().values
        curr_vals = current_df[feat].dropna().values
        
        # PSI
        psi = calculate_psi(base_vals, curr_vals)
        
        # KS-Test (two-sample test for distributional equality)
        ks_stat, p_val = ks_2samp(base_vals, curr_vals)
        
        # Severity status
        if psi >= 0.25:
            status = "CRITICAL_DRIFT"
        elif psi >= 0.10:
            status = "MODERATE_DRIFT"
        else:
            status = "STABLE"

        drift_records.append({
            "feature": feat,
            "psi": psi,
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 5),
            "status": status
        })

    drift_records = sorted(drift_records, key=lambda x: x["psi"], reverse=True)
    
    # Portfolio health summary
    n_critical = sum(1 for r in drift_records if r["status"] == "CRITICAL_DRIFT")
    n_moderate = sum(1 for r in drift_records if r["status"] == "MODERATE_DRIFT")
    
    if n_critical > 0:
        overall_health = "ACTION_REQUIRED"
    elif n_moderate > 0:
        overall_health = "MONITOR_CLOSELY"
    else:
        overall_health = "HEALTHY"

    return {
        "overall_portfolio_status": overall_health,
        "critical_features_count": n_critical,
        "moderate_features_count": n_moderate,
        "feature_metrics": drift_records
    }

def generate_macro_shock_batch(baseline_df: pd.DataFrame, shock_severity: float = 0.5) -> pd.DataFrame:
    """
    Simulates a macroeconomic shock (e.g. inflation surge & credit squeeze).
    Increases revolving utilization, elevates DTI, and softens FICO scores.
    Used for live stress-testing demonstration.
    """
    df_shock = baseline_df.copy()
    
    # Stress transformations proportional to severity (0.0 to 1.0)
    df_shock["fico_score"] = np.clip(
        df_shock["fico_score"] - (shock_severity * 45 + np.random.normal(0, 10, len(df_shock))),
        300, 850
    ).astype(int)
    
    df_shock["revolving_utilization_ratio"] = np.clip(
        df_shock["revolving_utilization_ratio"] * (1.0 + shock_severity * 0.40),
        0.0, 1.20
    )
    
    df_shock["total_debt_to_income_ratio"] = np.clip(
        df_shock["total_debt_to_income_ratio"] * (1.0 + shock_severity * 0.35),
        0.05, 1.50
    )
    
    df_shock["rolling_avg_dpd_6m"] = np.clip(
        df_shock["rolling_avg_dpd_6m"] + (shock_severity * 15),
        0, 180
    )

    return df_shock

if __name__ == "__main__":
    test_path = MODELS_DIR / "holdout_test.parquet"
    if test_path.exists():
        df_base = pd.read_parquet(test_path)
        df_stressed = generate_macro_shock_batch(df_base, shock_severity=0.8)
        
        num_features = ["fico_score", "revolving_utilization_ratio", "total_debt_to_income_ratio", "annual_income"]
        report = evaluate_portfolio_drift(df_base, df_stressed, num_features)
        print("Portfolio Drift Report under 80% Macro Shock:")
        print(json.dumps(report, indent=2))
