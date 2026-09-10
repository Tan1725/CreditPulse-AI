"""
CreditPulse-AI Master Pipeline Orchestrator
Executes the full end-to-end workflow:
1. Seeds DuckDB Relational Warehouse (10,000 borrowers, 120,000 payment records)
2. Compiles SQL Feature Engineering Pipeline (Window functions, CTEs)
3. Trains & Benchmarks ML Models (Logistic Regression, Random Forest, XGBoost)
4. Calibrates Default Probabilities (Platt / Isotonic)
5. Solves Cost-Sensitive Decision Threshold (Profit maximization vs 0.5)
6. Computes SHAP Global Attribution & Adverse Action Profiles
"""

import sys
import time
from pathlib import Path

# Fix Windows console encoding for Unicode/emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_manager import populate_database, DB_PATH
from src.database.query_runner import load_modeling_dataset
from src.ml.train import run_training_benchmark
from src.ml.calibration import calibrate_best_model
from src.ml.optimizer import optimize_decision_threshold
from src.ml.explainability import compute_shap_explanations

def run_all(force_reseed: bool = False):
    start_time = time.time()
    print("=" * 70)
    print(">> STARTING CREDITPULSE-AI END-TO-END PIPELINE")
    print("=" * 70)

    # 1. Database Check & Seeding
    if force_reseed or not DB_PATH.exists():
        print("\n[Step 1/6] Seeding DuckDB Relational Data Warehouse...")
        populate_database(n_customers=10000, force_recreate=True)
    else:
        print("\n[Step 1/6] Existing DuckDB warehouse detected at:", DB_PATH)

    # 2. SQL Feature Engineering Verification
    print("\n[Step 2/6] Executing SQL Feature Engineering Pipeline...")
    df_features = load_modeling_dataset()
    print(f"   Feature View Materialized: {df_features.shape[0]:,} rows, {df_features.shape[1]} columns")
    print(f"   Portfolio Default Rate:   {df_features['is_default'].mean():.2%}")

    # 3. Model Training & Benchmarking
    print("\n[Step 3/6] Benchmarking ML Classifiers (LR, RF, XGBoost)...")
    benchmark_results = run_training_benchmark()

    # 4. Probability Calibration
    print("\n[Step 4/6] Calibrating Probabilities for Empirical Risk Reliability...")
    cal_report = calibrate_best_model()

    # 5. Cost-Sensitive Optimization
    print("\n[Step 5/6] Optimizing Financial Decision Cutoff (tau*)...")
    opt_summary = optimize_decision_threshold()

    # 6. Explainable AI (SHAP)
    print("\n[Step 6/6] Computing SHAP Global & Adverse Action Attribution...")
    compute_shap_explanations()

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS!")
    print("=" * 70)
    print("Summary Highlights:")
    print(f" * Optimal Policy Threshold (tau*): {opt_summary['optimal_threshold']:.2f}")
    print(f" * Incremental Profit Gained:       +${opt_summary['incremental_profit_dollars']:,.2f} (+{opt_summary['percentage_profit_improvement']}%)")
    print(f" * Brier Score Calibration Gain:    +{cal_report['brier_improvement_pct']}%")
    print("=" * 70)

if __name__ == "__main__":
    run_all()
