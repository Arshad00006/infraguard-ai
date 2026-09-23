"""
Model training, prediction and SHAP explanation utilities for INFRAguard AI.
"""

import os
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import shap

FEATURE_COLS = [
    "original_cost_cr",
    "revised_cost_cr",
    "expenditure_cr",
    "physical_progress_pct",
    "planned_duration_months",
    "milestone_slip_months",
    "spend_deviation_pct",
    "cost_overrun_pct",
    "progress_variance",
]

TARGET_COLS = ["cost_risk", "delay_risk", "overall_risk"]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and clean feature columns."""
    X = df[FEATURE_COLS].copy()
    X = X.fillna(X.median(numeric_only=True))
    return X


def train_models(df: pd.DataFrame, model_dir: str = "models"):
    """Train CatBoost models for cost, delay and overall risk."""
    os.makedirs(model_dir, exist_ok=True)
    X = prepare_features(df)
    results = {}

    for target in TARGET_COLS:
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        model = CatBoostClassifier(
            iterations=120,
            depth=5,
            learning_rate=0.08,
            loss_function="Logloss",
            verbose=0,
            random_seed=42,
            eval_metric="AUC",
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=20)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 3),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 3),
            "auc": round(float(roc_auc_score(y_test, y_proba)), 3) if len(np.unique(y_test)) > 1 else 0.5,
        }
        results[target] = metrics

        model_path = os.path.join(model_dir, f"{target}_model.cbm")
        model.save_model(model_path)
        print(f"Saved {model_path} | Metrics: {metrics}")

    with open(os.path.join(model_dir, "feature_cols.json"), "w") as f:
        json.dump(FEATURE_COLS, f)

    return results


def load_models(model_dir: str = "models"):
    """Load trained models."""
    models = {}
    for target in TARGET_COLS:
        path = os.path.join(model_dir, f"{target}_model.cbm")
        if os.path.exists(path):
            m = CatBoostClassifier()
            m.load_model(path)
            models[target] = m
    return models


def predict_risks(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Add risk probability columns to dataframe."""
    X = prepare_features(df)
    out = df.copy()

    for target, model in models.items():
        proba = model.predict_proba(X)[:, 1]
        out[f"{target}_prob"] = np.round(proba * 100, 1)
        out[f"{target}_flag"] = (proba >= 0.5).astype(int)

    if "cost_risk_prob" in out.columns and "delay_risk_prob" in out.columns:
        out["risk_score"] = np.round(
            0.45 * out["cost_risk_prob"] + 0.55 * out["delay_risk_prob"], 1
        )
    return out


def get_shap_explanation(model, X_row: pd.DataFrame, feature_names: list):
    """Compute SHAP values for a single project."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    if hasattr(sv, "ndim") and sv.ndim > 1:
        sv = sv[0]

    contrib = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv,
        "feature_value": X_row.iloc[0].values,
    })
    contrib["abs_shap"] = contrib["shap_value"].abs()
    contrib = contrib.sort_values("abs_shap", ascending=False).reset_index(drop=True)
    return contrib


def simulate_intervention(row: pd.Series, changes: dict, models: dict) -> dict:
    """What-if simulation: apply changes and re-predict risk."""
    modified = row.copy()
    for k, v in changes.items():
        if k in modified.index:
            modified[k] = v

    if "revised_cost_cr" in changes or "original_cost_cr" in changes:
        modified["cost_overrun_pct"] = (
            (modified["revised_cost_cr"] - modified["original_cost_cr"])
            / max(modified["original_cost_cr"], 1e-6)
        ) * 100

    X = prepare_features(pd.DataFrame([modified]))
    results = {"original": {}, "simulated": {}}

    for target, model in models.items():
        orig_prob = model.predict_proba(prepare_features(pd.DataFrame([row])))[0, 1] * 100
        new_prob = model.predict_proba(X)[0, 1] * 100
        results["original"][target] = round(float(orig_prob), 1)
        results["simulated"][target] = round(float(new_prob), 1)

    return results
