"""
features/slot_features.py
=========================
Compute slot-calibrated (hour-of-week) z-scores and aggregate features.

The key idea: compare each reading to "what's normal at this hour-of-week"
using 168 slots (24h × 7 days). This removes daily/weekly seasonality.

Usage:
    from src.features.slot_features import build_features
    features, slot_z, model_feature_names = build_features(df_clean, numeric_cols)
"""

import pandas as pd
import numpy as np


def compute_slot_z_scores(
    df: pd.DataFrame,
    numeric_cols: list[str],
    z_cap: float = 20.0,
) -> pd.DataFrame:
    """
    Compute IQR-based robust z-scores relative to each hour-of-week slot.

    For each company column and each timestamp:
        z = (value - slot_median) / slot_IQR

    Capped at ±z_cap to prevent astronomical values from near-zero IQR columns.
    """
    slot_z = pd.DataFrame(index=df.index)

    for col in numeric_cols:
        grp = df.groupby("slot")[col]
        med = grp.transform("median")
        q25 = grp.transform(lambda x: np.percentile(x, 25))
        q75 = grp.transform(lambda x: np.percentile(x, 75))
        iqr = q75 - q25

        # Fallback to std if IQR is zero
        scale = iqr.copy()
        zero_mask = scale < 1e-6
        if zero_mask.any():
            scale[zero_mask] = grp.transform("std")[zero_mask]
        scale = scale.clip(lower=1e-6)

        slot_z[col] = ((df[col] - med) / scale).clip(-z_cap, z_cap)

    slot_z = slot_z.replace([np.inf, -np.inf], 0).fillna(0)
    return slot_z


def build_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    z_cap: float = 20.0,
    z_mild: float = 3.0,
    z_extreme: float = 5.0,
    rolling_window: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Build the full feature matrix from cleaned data.

    Returns:
        features: DataFrame with timestamp + all features + total_power
        slot_z:   DataFrame of per-company slot z-scores (for contributor analysis)
        model_feature_names: List of feature column names for the model
    """
    print("Computing slot-calibrated z-scores...")
    slot_z = compute_slot_z_scores(df, numeric_cols, z_cap=z_cap)

    print("Building aggregate features...")
    features = pd.DataFrame(index=df.index)
    features["timestamp"] = df["timestamp"]
    features["slot"] = df["slot"]

    # 1. Mean absolute z: overall "weirdness" across all companies
    features["mean_abs_slot_z"] = slot_z[numeric_cols].abs().mean(axis=1)

    # 2. Max absolute z: worst single company
    features["max_slot_z"] = slot_z[numeric_cols].abs().max(axis=1)

    # 3-4. Count of flagged companies
    features["n_flagged_3z"] = (slot_z[numeric_cols].abs() > z_mild).sum(axis=1)
    features["n_flagged_5z"] = (slot_z[numeric_cols].abs() > z_extreme).sum(axis=1)

    # 5. Total system power + its slot z-score
    df["total_power"] = df[numeric_cols].sum(axis=1)
    total_med = df.groupby("slot")["total_power"].transform("median")
    total_iqr = (
        df.groupby("slot")["total_power"]
        .transform(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        .clip(lower=1e-6)
    )
    features["total_power"] = df["total_power"]
    features["total_power_slot_z"] = ((df["total_power"] - total_med) / total_iqr).clip(
        -z_cap, z_cap
    )

    # 6. Absolute one-step change
    features["delta_total"] = df["total_power"].diff().abs().fillna(0)

    # 7. Rolling volatility
    features["rolling_vol_6h"] = (
        df["total_power"].rolling(rolling_window, min_periods=1).std().fillna(0)
    )

    # 8. HHI (Herfindahl-Hirschman Index) — market concentration
    row_sums = df[numeric_cols].sum(axis=1).replace(0, 1)
    shares = df[numeric_cols].div(row_sums, axis=0).fillna(0)
    features["hhi"] = (shares**2).sum(axis=1)

    # 9. Top-5 company share
    features["top5_share"] = np.sort(shares.values, axis=1)[:, -5:].sum(axis=1)

    # Model feature names
    model_feature_names = [
        "mean_abs_slot_z",
        "max_slot_z",
        "n_flagged_3z",
        "n_flagged_5z",
        "total_power_slot_z",
        "delta_total",
        "rolling_vol_6h",
        "hhi",
        "top5_share",
    ]

    # Clean any remaining inf/nan
    for col in model_feature_names:
        features[col] = features[col].replace([np.inf, -np.inf], 0).fillna(0)

    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Model features: {model_feature_names}")
    return features, slot_z, model_feature_names
