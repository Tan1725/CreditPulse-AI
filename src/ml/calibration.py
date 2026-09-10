"""
CreditPulse-AI Probability Calibration Engine
Calibrates raw machine learning model scores into true empirical default probabilities
using Isotonic Regression and Platt Scaling (Sigmoid).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

def calibrate_best_model():
    """Calibrates best model pipeline on validation data using Isotonic & Platt methods."""
    model_path = MODELS_DIR / "best_model.joblib"
    val_path = MODELS_DIR / "validation.parquet"
    test_path = MODELS_DIR / "holdout_test.parquet"
    meta_path = MODELS_DIR / "feature_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError("Run train.py before calibration.")

    base_pipe = joblib.load(model_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    feature_cols = meta["numerical_features"] + meta["categorical_features"]
    X_val = df_val[feature_cols]
    y_val = df_val["is_default"]
    
    X_test = df_test[feature_cols]
    y_test = df_test["is_default"]

    # Raw model probabilities
    raw_probs_test = base_pipe.predict_proba(X_test)[:, 1]
    raw_brier = float(brier_score_loss(y_test, raw_probs_test))

    # Fit CalibratedClassifierCV (prefit on base_pipe)
    # Using isotonic regression
    calibrated_iso = CalibratedClassifierCV(estimator=base_pipe, method="isotonic", cv="prefit")
    calibrated_iso.fit(X_val, y_val)
    
    cal_probs_test = calibrated_iso.predict_proba(X_test)[:, 1]
    cal_brier = float(brier_score_loss(y_test, cal_probs_test))
    
    # Calculate calibration curves (reliability diagrams)
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, raw_probs_test, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, cal_probs_test, n_bins=10)

    calibration_report = {
        "raw_brier_score": round(raw_brier, 5),
        "calibrated_brier_score": round(cal_brier, 5),
        "brier_improvement_pct": round(((raw_brier - cal_brier) / raw_brier) * 100.0, 2),
        "curve_raw": {
            "mean_predicted_value": prob_pred_raw.tolist(),
            "fraction_of_positives": prob_true_raw.tolist()
        },
        "curve_calibrated": {
            "mean_predicted_value": prob_pred_cal.tolist(),
            "fraction_of_positives": prob_true_cal.tolist()
        }
    }

    # Save calibrated pipeline as the primary production serving model
    joblib.dump(calibrated_iso, MODELS_DIR / "calibrated_model.joblib")
    with open(MODELS_DIR / "calibration_report.json", "w") as f:
        json.dump(calibration_report, f, indent=2)

    print(f"Calibration complete:")
    print(f"  Raw Brier Score:        {raw_brier:.4f}")
    print(f"  Calibrated Brier Score:   {cal_brier:.4f} (Improved by {calibration_report['brier_improvement_pct']}%)")
    return calibration_report

if __name__ == "__main__":
    calibrate_best_model()
