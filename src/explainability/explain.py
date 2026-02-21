"""
explainability/explain.py
=========================
SHAP and LIME explanations for anomaly detections.

Usage:
    from src.explainability.explain import run_shap, run_lime
    shap_values = run_shap(iforest, X_scaled, model_features, features)
    run_lime(iforest, X_scaled, model_features, features)
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def run_shap(
    iforest,
    X_scaled: np.ndarray,
    model_features: list[str],
    features: pd.DataFrame,
    sample_size: int = 5000,
    top_k_anomalies: int = 5,
    output_dir: str = "outputs/figures",
):
    """
    Compute SHAP values and generate:
      - Summary plot (global feature importance)
      - Waterfall plots for top-K anomalies
      - Dependence plot for top feature

    Returns shap_values array.
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Computing SHAP values...")

    explainer = shap.TreeExplainer(iforest)

    # Sample for speed
    n = min(sample_size, len(X_scaled))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_scaled), n, replace=False)
    X_sample = X_scaled[idx]

    shap_values = explainer.shap_values(X_sample)
    print(f"  SHAP computed for {n} samples")

    # --- Summary plot ---
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=model_features, show=False)
    plt.title("SHAP Summary — Which Features Drive Anomaly Scores?")
    plt.tight_layout()
    path = f"{output_dir}/shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- Waterfall plots for top anomalies ---
    mask = features["consensus"] == 1
    top_indices = (
        features[mask].sort_values("if_score", ascending=False).index[:top_k_anomalies]
    )

    for rank, aidx in enumerate(top_indices, 1):
        ts = features.loc[aidx, "timestamp"]
        reasons = features.loc[aidx, "reasons"]
        x_single = X_scaled[aidx : aidx + 1]
        sv = explainer.shap_values(x_single)

        # Handle base_values — convert to scalar for newer SHAP versions
        bv = explainer.expected_value
        if hasattr(bv, "__len__"):
            bv = float(bv[0]) if len(bv) > 0 else float(bv)
        else:
            bv = float(bv)

        plt.figure(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv[0],
                base_values=bv,
                data=x_single[0],
                feature_names=model_features,
            ),
            show=False,
        )
        plt.title(f"SHAP Waterfall — Anomaly #{rank}: {ts}\n{reasons}", fontsize=11)
        plt.tight_layout()
        path = f"{output_dir}/shap_waterfall_{rank}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

    # --- Dependence plot ---
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        "mean_abs_slot_z",
        shap_values,
        X_sample,
        feature_names=model_features,
        show=False,
    )
    plt.title("SHAP Dependence — mean_abs_slot_z")
    plt.tight_layout()
    path = f"{output_dir}/shap_dependence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    return shap_values


def run_lime(
    iforest,
    X_scaled: np.ndarray,
    model_features: list[str],
    features: pd.DataFrame,
    top_k_anomalies: int = 5,
    output_dir: str = "outputs/figures",
):
    """
    Generate LIME explanations for top-K anomalies.
    Saves bar chart PNGs and prints feature contributions.
    """
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        print("LIME not installed. Run: pip install lime")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Wrapper function for LIME
    mn, mx = features["if_score"].min(), features["if_score"].max()

    def predict_proba(X):
        raw = -iforest.score_samples(X)
        norm = (raw - mn) / (mx - mn + 1e-10)
        return np.column_stack([1 - norm, norm])

    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        X_scaled,
        feature_names=model_features,
        class_names=["Normal", "Anomaly"],
        mode="classification",
    )

    mask = features["consensus"] == 1
    top_indices = (
        features[mask].sort_values("if_score", ascending=False).index[:top_k_anomalies]
    )

    for rank, aidx in enumerate(top_indices, 1):
        ts = features.loc[aidx, "timestamp"]
        reasons = features.loc[aidx, "reasons"]

        exp = lime_exp.explain_instance(
            X_scaled[aidx],
            predict_proba,
            num_features=len(model_features),
            labels=(1,),
        )

        fig = exp.as_pyplot_figure(label=1)
        fig.set_size_inches(10, 5)
        plt.title(f"LIME — Anomaly #{rank}: {ts}\n{reasons}", fontsize=11)
        plt.tight_layout()
        path = f"{output_dir}/lime_anomaly_{rank}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\n  Anomaly #{rank}: {ts}")
        print(f"  Reasons: {reasons}")
        print(f"  LIME contributions:")
        for feat, weight in exp.as_list(label=1):
            arrow = "→ ANOMALY" if weight > 0 else "→ normal"
            print(f"    {feat}: {weight:+.4f} {arrow}")

        print(f"  Saved: {path}")
