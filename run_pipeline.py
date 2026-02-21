#!/usr/bin/env python3
"""
run_pipeline.py
===============
Main entry point. Runs the full anomaly detection pipeline end-to-end.

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-shap --skip-lime   # faster, no explainability
    python run_pipeline.py --data path/to/your.csv    # custom data path
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.clean import load_and_clean
from src.features.slot_features import build_features
from src.models.detectors import run_detectors
from src.models.reasons import assign_reasons, get_top_contributors
from src.visualization.dashboard import plot_dashboard, plot_deepdive


def main():
    parser = argparse.ArgumentParser(
        description="Colombia Power Grid Anomaly Detection"
    )
    parser.add_argument(
        "--data",
        default="data/raw/sold_power_wide_2020_2023.csv",
        help="Path to raw CSV",
    )
    parser.add_argument(
        "--skip-shap", action="store_true", help="Skip SHAP explanations"
    )
    parser.add_argument(
        "--skip-lime", action="store_true", help="Skip LIME explanations"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("COLOMBIA POWER GRID — INTERPRETABLE ANOMALY DETECTION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. LOAD & CLEAN
    # ------------------------------------------------------------------
    df_clean, numeric_cols = load_and_clean(
        raw_path=args.data,
        missing_threshold=50.0,
        save_path="data/processed/cleaned.csv",
    )

    # ------------------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # ------------------------------------------------------------------
    features, slot_z, model_features = build_features(df_clean, numeric_cols)

    # Save features
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    features.to_csv("data/processed/features.csv", index=False)
    print(f"Saved: data/processed/features.csv")

    # ------------------------------------------------------------------
    # 3. ANOMALY DETECTION
    # ------------------------------------------------------------------
    features, iforest, ocsvm, scaler = run_detectors(features, model_features)

    # ------------------------------------------------------------------
    # 4. REASON LABELS + TOP CONTRIBUTORS
    # ------------------------------------------------------------------
    features = assign_reasons(features)
    report_df = get_top_contributors(features, slot_z, df_clean, numeric_cols)

    # Print top anomalies
    print(f"\n{'=' * 80}")
    print(f"TOP 10 ANOMALIES")
    print(f"{'=' * 80}")
    for _, row in report_df.head(10).iterrows():
        print(f"\n  📍 {row['timestamp']} | IF={row['if_score']:.3f}")
        print(f"     Why: {row['reasons']}")
        print(f"     Top: {row['top_contributor_1']}")

    # ------------------------------------------------------------------
    # 5. SAVE OUTPUTS
    # ------------------------------------------------------------------
    Path("outputs/scores").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    features.to_csv("outputs/scores/anomaly_features_full.csv", index=False)
    report_df.to_csv("outputs/reports/anomaly_report.csv", index=False)
    print(f"\nSaved: outputs/scores/anomaly_features_full.csv")
    print(f"Saved: outputs/reports/anomaly_report.csv")

    # ------------------------------------------------------------------
    # 6. VISUALIZATIONS
    # ------------------------------------------------------------------
    plot_dashboard(features, model_features)
    plot_deepdive(features, report_df)

    # ------------------------------------------------------------------
    # 7. SHAP + LIME (optional)
    # ------------------------------------------------------------------
    X_scaled = scaler.transform(features[model_features].values)

    if not args.skip_shap:
        from src.explainability.explain import run_shap

        run_shap(iforest, X_scaled, model_features, features)

    if not args.skip_lime:
        from src.explainability.explain import run_lime

        run_lime(iforest, X_scaled, model_features, features)

    # ------------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------------
    n_both = features["consensus"].sum()
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE!")
    print(f"  Total timestamps: {len(features)}")
    print(f"  Consensus anomalies: {n_both}")
    print(f"  Report: outputs/reports/anomaly_report.csv")
    print(f"  Dashboard: outputs/figures/anomaly_dashboard.png")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
