"""
CreditPulse-AI Model Explainability Engine (SHAP)
Provides global feature attribution and individual borrower adverse action explanations
required for fair lending regulatory compliance (ECOA / FCRA).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

def compute_shap_explanations(max_samples: int = 500):
    """Computes global SHAP values and pre-computes feature importances."""
    base_model_path = MODELS_DIR / "best_model.joblib"
    test_path = MODELS_DIR / "holdout_test.parquet"
    meta_path = MODELS_DIR / "feature_metadata.json"

    base_pipe = joblib.load(base_model_path)
    df_test = pd.read_parquet(test_path)
    
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_cols = meta["numerical_features"] + meta["categorical_features"]
    preprocessor = base_pipe.named_steps["preprocessor"]
    clf = base_pipe.named_steps["classifier"]

    # Transform a sample of test data
    sample_df = df_test[feature_cols].head(max_samples)
    X_transformed = preprocessor.transform(sample_df)
    feature_names = preprocessor.get_feature_names_out()

    # Use TreeExplainer if tree model, else general Explainer
    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)
    except Exception:
        explainer = shap.Explainer(clf.predict_proba, X_transformed)
        shap_values = explainer(X_transformed).values
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

    # Handle binary classification shapes (some SHAP versions return list or 2D array)
    if isinstance(shap_values, list):
        shap_matrix = shap_values[1]  # positive class (default)
    elif len(shap_values.shape) == 3:
        shap_matrix = shap_values[:, :, 1]
    else:
        shap_matrix = shap_values

    # Mean absolute SHAP per feature
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    global_importance = []
    for name, score in zip(feature_names, mean_abs_shap):
        clean_name = name.replace("num__", "").replace("cat__", "")
        global_importance.append({"feature": clean_name, "importance": round(float(score), 4)})
    
    global_importance = sorted(global_importance, key=lambda x: x["importance"], reverse=True)

    # Save explainer object and feature names
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")
    joblib.dump(feature_names, MODELS_DIR / "transformed_feature_names.joblib")
    
    with open(MODELS_DIR / "global_feature_importance.json", "w") as f:
        json.dump(global_importance, f, indent=2)

    print(f"[SUCCESS] SHAP explanations computed. Top 5 risk drivers across portfolio:")
    for item in global_importance[:5]:
        print(f"   - {item['feature']}: {item['importance']}")

    return global_importance

def explain_single_applicant(applicant_dict: dict) -> list:
    """
    Generates regulatory adverse action reasons for a single applicant.
    Returns sorted list of features that pushed risk higher or lower.
    """
    base_model_path = MODELS_DIR / "best_model.joblib"
    meta_path = MODELS_DIR / "feature_metadata.json"
    
    base_pipe = joblib.load(base_model_path)
    preprocessor = base_pipe.named_steps["preprocessor"]
    clf = base_pipe.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    df_single = pd.DataFrame([applicant_dict])
    X_trans = preprocessor.transform(df_single)

    if hasattr(clf, "coef_"):
        # Exact linear log-odds contribution (sub-millisecond computation)
        vals = (clf.coef_[0] * X_trans[0]).tolist()
    else:
        try:
            explainer = joblib.load(MODELS_DIR / "shap_explainer.joblib")
            shap_vals = explainer.shap_values(X_trans)
            if isinstance(shap_vals, list):
                vals = shap_vals[1][0]
            elif len(shap_vals.shape) == 3:
                vals = shap_vals[0, :, 1]
            else:
                vals = shap_vals[0]
        except Exception:
            vals = np.zeros(len(feature_names))

    adverse_reasons = []
    for fname, val in zip(feature_names, vals):
        clean_name = fname.replace("num__", "").replace("cat__", "")
        adverse_reasons.append({
            "feature": clean_name,
            "shap_value": round(float(val), 4),
            "impact": "Increases Risk" if val > 0 else "Decreases Risk"
        })
    
    # Sort by magnitude of contribution to risk
    adverse_reasons = sorted(adverse_reasons, key=lambda x: abs(x["shap_value"]), reverse=True)
    return adverse_reasons

if __name__ == "__main__":
    compute_shap_explanations()
