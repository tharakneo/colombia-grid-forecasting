"""
models/detectors.py
===================
Anomaly detection models: Isolation Forest, One-Class SVM, and consensus logic.

Usage:
    from src.models.detectors import run_detectors
    features = run_detectors(features, model_feature_names)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


def run_detectors(
    features: pd.DataFrame,
    model_feature_names: list[str],
    if_params: dict = None,
    ocsvm_params: dict = None,
) -> tuple[pd.DataFrame, IsolationForest, OneClassSVM, StandardScaler]:
    """
    Run Isolation Forest and One-Class SVM on the feature matrix.

    Adds columns: if_score, if_label, ocsvm_score, ocsvm_label,
                  if_score_norm, ocsvm_score_norm, consensus

    Returns:
        features: Updated DataFrame with scores
        iforest: Trained Isolation Forest model (for SHAP)
        ocsvm: Trained OCSVM model
        scaler: Fitted StandardScaler (for LIME)
    """
    if if_params is None:
        if_params = {"n_estimators": 300, "contamination": 0.01, "random_state": 42}
    if ocsvm_params is None:
        ocsvm_params = {"kernel": "rbf", "gamma": "scale", "nu": 0.01}

    X = features[model_feature_names].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Isolation Forest ---
    print("Training Isolation Forest...")
    iforest = IsolationForest(n_jobs=-1, **if_params)
    iforest.fit(X_scaled)
    features["if_score"] = -iforest.score_samples(X_scaled)
    features["if_label"] = iforest.predict(X_scaled)

    # --- One-Class SVM ---
    print("Training One-Class SVM...")
    ocsvm = OneClassSVM(**ocsvm_params)
    ocsvm.fit(X_scaled)
    features["ocsvm_score"] = -ocsvm.decision_function(X_scaled)
    features["ocsvm_label"] = ocsvm.predict(X_scaled)

    # --- Normalize scores to 0–1 ---
    for col in ["if_score", "ocsvm_score"]:
        mn, mx = features[col].min(), features[col].max()
        features[f"{col}_norm"] = (features[col] - mn) / (mx - mn + 1e-10)

    # --- Consensus ---
    features["consensus"] = (
        (features["if_label"] == -1) & (features["ocsvm_label"] == -1)
    ).astype(int)

    n_if = (features["if_label"] == -1).sum()
    n_ocsvm = (features["ocsvm_label"] == -1).sum()
    n_both = features["consensus"].sum()

    print(f"\nResults:")
    print(f"  IF anomalies:    {n_if} ({n_if / len(features) * 100:.2f}%)")
    print(f"  OCSVM anomalies: {n_ocsvm} ({n_ocsvm / len(features) * 100:.2f}%)")
    print(f"  Consensus:       {n_both} ({n_both / len(features) * 100:.2f}%)")

    return features, iforest, ocsvm, scaler
