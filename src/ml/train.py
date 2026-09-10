"""
CreditPulse-AI Model Training & Benchmarking Pipeline
Trains, cross-validates, and compares Logistic Regression, Random Forest,
and XGBoost with probability calibration and metric tracking.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, brier_score_loss
)

from src.database.query_runner import load_modeling_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

NUMERICAL_FEATURES = [
    "age", "annual_income", "employment_years", "fico_score",
    "open_credit_lines", "bankruptcies", "inquiries_last_6m",
    "revolving_utilization_ratio", "total_debt_to_income_ratio",
    "payment_to_income_ratio", "loan_amount", "interest_rate",
    "term_months", "rolling_avg_dpd_6m", "prior_max_dpd_12m",
    "prior_delinquent_months", "rolling_payment_ratio_3m"
]

CATEGORICAL_FEATURES = ["home_ownership", "education_level", "loan_intent"]
TARGET = "is_default"

def prepare_pipeline():
    """Build preprocessing pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def evaluate_model(model, X_test, y_test) -> dict:
    """Computes comprehensive classification and probability metrics."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    return {
        "roc_auc": round(float(roc_auc_score(y_test, y_pred_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_pred_proba)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_pred_proba)), 4)
    }

def run_training_benchmark():
    """Executes multi-model training and logs comparison benchmark."""
    print("Loading feature warehouse dataset...")
    df = load_modeling_dataset()
    
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    
    # Train / Val / Test split (70% train, 15% validation, 15% holdout test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"Dataset split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    neg_count, pos_count = np.bincount(y_train)
    imbalance_ratio = round(neg_count / pos_count, 2)
    print(f"Class distribution: Non-defaults={neg_count}, Defaults={pos_count} (Ratio {imbalance_ratio}:1)")

    # Candidate models
    models = {
        "Logistic_Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=imbalance_ratio, eval_metric="logloss", random_state=42
        )
    }

    results = {}
    fitted_pipelines = {}

    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        pipe = Pipeline(steps=[
            ("preprocessor", prepare_pipeline()),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)
        
        metrics_val = evaluate_model(pipe, X_val, y_val)
        metrics_test = evaluate_model(pipe, X_test, y_test)
        
        print(f"Validation Metrics: ROC-AUC={metrics_val['roc_auc']}, PR-AUC={metrics_val['pr_auc']}, Brier={metrics_val['brier_score']}")
        print(f"Test Holdout:      ROC-AUC={metrics_test['roc_auc']}, PR-AUC={metrics_test['pr_auc']}, F1={metrics_test['f1_score']}")
        
        results[name] = {
            "validation": metrics_val,
            "test": metrics_test
        }
        fitted_pipelines[name] = pipe

    # Choose winner based on Validation PR-AUC and ROC-AUC
    winner_name = max(results.keys(), key=lambda k: results[k]["validation"]["roc_auc"] + results[k]["validation"]["pr_auc"])
    print(f"\n[WINNER] Best Model Selected: {winner_name}")
    best_pipeline = fitted_pipelines[winner_name]

    # Save artifacts
    joblib.dump(best_pipeline, MODELS_DIR / "best_model.joblib")
    joblib.dump(fitted_pipelines["Logistic_Regression"], MODELS_DIR / "baseline_model.joblib")
    
    # Save test dataset for calibration and threshold evaluation
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df.to_parquet(MODELS_DIR / "holdout_test.parquet", index=False)
    
    val_df = X_val.copy()
    val_df[TARGET] = y_val.values
    val_df.to_parquet(MODELS_DIR / "validation.parquet", index=False)

    metadata = {
        "best_model_name": winner_name,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "class_imbalance_ratio": imbalance_ratio,
        "sample_counts": {"train": len(X_train), "val": len(X_val), "test": len(X_test)}
    }
    
    with open(MODELS_DIR / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(MODELS_DIR / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("[SUCCESS] Model artifacts successfully saved to:", MODELS_DIR)
    return results

if __name__ == "__main__":
    run_training_benchmark()
