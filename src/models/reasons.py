"""
models/reasons.py
=================
Assign interpretable reason labels and identify top contributing companies.

Usage:
    from src.models.reasons import assign_reasons, get_top_contributors
    features = assign_reasons(features)
    report_df = get_top_contributors(features, slot_z, df_clean, numeric_cols)
"""

import pandas as pd
import numpy as np


def assign_reasons(
    features: pd.DataFrame,
    system_z_thresh: float = 3.0,
    widespread_thresh: int = 10,
    extreme_z_thresh: float = 10.0,
    pct_delta: float = 99.0,
    pct_vol: float = 99.0,
    pct_hhi: float = 99.0,
) -> pd.DataFrame:
    """
    Assign human-readable reason labels to each consensus anomaly.

    Reasons:
        SYSTEM_SPIKE / SYSTEM_DROP — total power far from slot normal
        WIDESPREAD(N companies)    — many companies flagged simultaneously
        SUDDEN_CHANGE              — large hour-to-hour jump (top 1%)
        HIGH_VOLATILITY            — unstable over past 6 hours (top 1%)
        MARKET_CONCENTRATION       — sales concentrated in few companies (top 1%)
        EXTREME_SINGLE_COMPANY     — one company has |z| > 10
        MULTI_FEATURE_MILD         — no single dominant reason
    """
    p99_delta = features["delta_total"].quantile(pct_delta / 100)
    p99_vol = features["rolling_vol_6h"].quantile(pct_vol / 100)
    p99_hhi = features["hhi"].quantile(pct_hhi / 100)

    def _label(row):
        reasons = []
        if abs(row["total_power_slot_z"]) > system_z_thresh:
            reasons.append(
                "SYSTEM_SPIKE" if row["total_power_slot_z"] > 0 else "SYSTEM_DROP"
            )
        if row["n_flagged_3z"] > widespread_thresh:
            reasons.append(f'WIDESPREAD({int(row["n_flagged_3z"])} companies)')
        if row["delta_total"] > p99_delta:
            reasons.append("SUDDEN_CHANGE")
        if row["rolling_vol_6h"] > p99_vol:
            reasons.append("HIGH_VOLATILITY")
        if row["hhi"] > p99_hhi:
            reasons.append("MARKET_CONCENTRATION")
        if row["max_slot_z"] > extreme_z_thresh:
            reasons.append("EXTREME_SINGLE_COMPANY")
        if not reasons:
            reasons.append("MULTI_FEATURE_MILD")
        return " | ".join(reasons)

    mask = features["consensus"] == 1
    features["reasons"] = ""
    if mask.sum() > 0:
        features.loc[mask, "reasons"] = features.loc[mask].apply(_label, axis=1)

    # Print summary
    if mask.sum() > 0:
        reason_series = (
            features.loc[mask, "reasons"]
            .str.split(" \\| ")
            .explode()
            .str.replace(r"\(\d+ companies\)", "", regex=True)
            .str.strip()
        )
        print("\nAnomaly Reason Distribution:")
        print(reason_series.value_counts().to_string())

    return features


def get_top_contributors(
    features: pd.DataFrame,
    slot_z: pd.DataFrame,
    df_clean: pd.DataFrame,
    numeric_cols: list[str],
    top_k: int = 3,
) -> pd.DataFrame:
    """
    For each consensus anomaly, find the top-K companies with the largest |z|.

    Returns a report DataFrame sorted by IF score descending.
    """
    mask = features["consensus"] == 1
    results = []

    for idx in features[mask].index:
        row_z = slot_z.loc[idx, numeric_cols]
        top = row_z.abs().nlargest(top_k)

        contribs = {}
        for rank, col in enumerate(top.index, 1):
            z_val = row_z[col]
            raw_val = df_clean.loc[idx, col]
            direction = "↑ HIGH" if z_val > 0 else "↓ LOW"
            contribs[f"top_contributor_{rank}"] = (
                f"{col} {direction} (z={z_val:.1f}, val={raw_val:,.0f})"
            )

        results.append(
            {
                "timestamp": features.loc[idx, "timestamp"],
                "if_score": round(features.loc[idx, "if_score_norm"], 3),
                "ocsvm_score": round(features.loc[idx, "ocsvm_score_norm"], 3),
                "reasons": features.loc[idx, "reasons"],
                "n_companies_flagged": int(features.loc[idx, "n_flagged_3z"]),
                **contribs,
            }
        )

    report = pd.DataFrame(results).sort_values("if_score", ascending=False)
    return report
