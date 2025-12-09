#!/usr/bin/env python3
"""
Normalize sold power (2020-2023) with leak-free Z-score stats from 2020-2022.

Inputs:
  - sold_power_wide_2020_2023.csv   (must have a 'timestamp' column)

Outputs:
  - sold_power_wide_2020_2023_normalized.csv
  - sold_power_wide_normalization_params.csv

Run:
  python normalize_power.py
"""

import sys
import pandas as pd
import numpy as np

SRC_CSV = "sold_power_wide_2020_2023.csv"
OUT_NORM = "sold_power_wide_2020_2023_normalized.csv"
OUT_PARAMS = "sold_power_wide_normalization_params.csv"


def main():
    # 1) Load
    try:
        df = pd.read_csv(SRC_CSV, parse_dates=["timestamp"])
    except FileNotFoundError:
        print(f"ERROR: Cannot find {SRC_CSV} in the current folder.")
        sys.exit(1)
    except Exception as e:
        print("ERROR reading CSV:", e)
        sys.exit(1)

    # 2) Basic checks
    if "timestamp" not in df.columns:
        print("ERROR: CSV must contain a 'timestamp' column.")
        sys.exit(1)

    # 3) Split training window: 2020–2022 (leak-free stats)
    years = df["timestamp"].dt.year
    train_mask = (years >= 2020) & (years <= 2022)
    train = df.loc[train_mask].copy()
    if train.empty:
        print("ERROR: No rows found in years 2020–2022 for training statistics.")
        sys.exit(1)

    # 4) Identify numeric columns (exclude timestamp)
    series_cols = [c for c in df.columns if c != "timestamp"]

    # 5) Compute mean/std on training only
    mu = train[series_cols].mean()
    sd = train[series_cols].std(ddof=0)

    # Handle zero std (constant series) to avoid division by zero
    # If std == 0, we leave those columns as 0 after normalization.
    zero_std_cols = sd.index[sd == 0].tolist()
    if zero_std_cols:
        print(
            f"NOTE: {len(zero_std_cols)} column(s) have zero std in 2020–2022 and will be set to 0 in normalized data."
        )

    # 6) Apply Z-score with training mu, std
    norm = df.copy()
    # Avoid division by zero by replacing 0 std with 1 only for computation,
    # then we'll zero those columns explicitly.
    sd_safe = sd.replace(0, 1)
    norm[series_cols] = (norm[series_cols] - mu) / sd_safe

    # Force zero-std columns to 0
    if zero_std_cols:
        norm[zero_std_cols] = 0.0

    # 7) Save outputs
    norm.to_csv(OUT_NORM, index=False)
    params = pd.DataFrame({"mean": mu, "std": sd})
    params.to_csv(OUT_PARAMS)

    # 8) Print quick sanity summary
    print("\n=== Normalization complete ===")
    print(f"Created: {OUT_NORM}")
    print(f"Created: {OUT_PARAMS}")

    # Check overall stats after normalization (all years)
    norm_means = norm[series_cols].mean().mean()
    norm_stds = norm[series_cols].std(ddof=0).mean()
    print(f"Avg of column means (all years, normalized): {norm_means:.4f}")
    print(f"Avg of column stds  (all years, normalized): {norm_stds:.4f}")

    # Check 2023 drift (optional quick glance)
    test_2023 = norm.loc[df["timestamp"].dt.year == 2023, series_cols]
    if not test_2023.empty:
        print(f"2023 avg mean across columns (z): {test_2023.mean().mean():.4f}")
        print(f"2023 avg std  across columns (z): {test_2023.std(ddof=0).mean():.4f}")


if __name__ == "__main__":
    main()
