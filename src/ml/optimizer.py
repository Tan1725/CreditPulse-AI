"""
CreditPulse-AI Cost-Sensitive Decision Threshold Optimizer
Derives the mathematically optimal decision cutoff (tau*) that maximizes
net portfolio financial return rather than relying on arbitrary 0.5 cutoffs.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

def compute_financial_impact(y_true, y_prob, loan_amounts, interest_rates, terms_months,
                             lgd: float = 0.70, margin_factor: float = 0.80):
    """
    Computes economic outcome across threshold spectrum tau in [0.01, 0.95].
    
    LGD (Loss Given Default): percentage of principal unrecoverable on default (default 70%).
    margin_factor: fraction of total interest earned net of servicing/cost-of-funds (default 80%).
    """
    # Potential profit if borrower repays
    interest_revenues = loan_amounts * (interest_rates / 100.0) * (terms_months / 12.0) * margin_factor
    # Potential loss if borrower defaults
    default_losses = loan_amounts * lgd

    thresholds = np.linspace(0.02, 0.90, 89)
    simulation_records = []

    for tau in thresholds:
        # Approval rule: Approve if predicted probability of default is strictly below tau
        approved = (y_prob < tau)
        n_approved = int(np.sum(approved))
        approval_rate = round(float(np.mean(approved)), 4)
        
        if n_approved == 0:
            net_profit = 0.0
            approved_default_rate = 0.0
        else:
            # For approved loans:
            app_y = y_true[approved]
            app_rev = interest_revenues[approved]
            app_loss = default_losses[approved]
            
            # Profit = interest from good loans - loss from defaulted loans
            net_profit = float(np.sum(np.where(app_y == 0, app_rev, -app_loss)))
            approved_default_rate = round(float(np.mean(app_y)), 4)

        simulation_records.append({
            "threshold": round(float(tau), 2),
            "approval_rate": approval_rate,
            "approved_count": n_approved,
            "approved_default_rate": approved_default_rate,
            "net_profit": round(net_profit, 2)
        })

    df_sim = pd.DataFrame(simulation_records)
    
    # Identify optimal threshold maximizing net portfolio profit
    best_idx = df_sim["net_profit"].idxmax()
    best_row = df_sim.loc[best_idx].to_dict()
    
    # Compare with default naive 0.5 threshold
    idx_naive = (df_sim["threshold"] - 0.50).abs().idxmin()
    naive_row = df_sim.loc[idx_naive].to_dict()
    
    incremental_profit = best_row["net_profit"] - naive_row["net_profit"]
    pct_profit_gain = (incremental_profit / max(abs(naive_row["net_profit"]), 1.0)) * 100.0

    summary = {
        "optimal_threshold": best_row["threshold"],
        "optimal_profit": best_row["net_profit"],
        "optimal_approval_rate": best_row["approval_rate"],
        "optimal_approved_default_rate": best_row["approved_default_rate"],
        
        "naive_threshold": 0.50,
        "naive_profit": naive_row["net_profit"],
        "naive_approval_rate": naive_row["approval_rate"],
        "naive_approved_default_rate": naive_row["approved_default_rate"],
        
        "incremental_profit_dollars": round(incremental_profit, 2),
        "percentage_profit_improvement": round(pct_profit_gain, 2),
        "simulation_curve": simulation_records
    }
    
    return summary

def optimize_decision_threshold():
    """Runs threshold optimization using the calibrated model and holdout test set."""
    model_path = MODELS_DIR / "calibrated_model.joblib"
    test_path = MODELS_DIR / "holdout_test.parquet"
    meta_path = MODELS_DIR / "feature_metadata.json"

    model = joblib.load(model_path)
    df_test = pd.read_parquet(test_path)
    
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_cols = meta["numerical_features"] + meta["categorical_features"]
    y_test = df_test["is_default"].values
    y_prob = model.predict_proba(df_test[feature_cols])[:, 1]

    summary = compute_financial_impact(
        y_true=y_test,
        y_prob=y_prob,
        loan_amounts=df_test["loan_amount"].values,
        interest_rates=df_test["interest_rate"].values,
        terms_months=df_test["term_months"].values
    )

    with open(MODELS_DIR / "threshold_optimization.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Cost-Sensitive Decision Threshold Optimization Results ---")
    print(f"Optimal Cutoff (tau*):          {summary['optimal_threshold']:.2f}")
    print(f"Net Profit at Optimal Cutoff:   ${summary['optimal_profit']:,.2f}")
    print(f"Approval Rate at Optimal Cutoff:{summary['optimal_approval_rate']:.1%}")
    print(f"Net Profit at Naive 0.50 Cutoff:${summary['naive_profit']:,.2f}")
    print(f"Incremental Profit Gained:      +${summary['incremental_profit_dollars']:,.2f} (+{summary['percentage_profit_improvement']}%)")
    
    return summary

if __name__ == "__main__":
    optimize_decision_threshold()
