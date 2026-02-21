"""
preprocessing/clean.py
======================
Load raw CSV, drop high-missing columns, impute, save cleaned version.

Usage:
    from src.preprocessing.clean import load_and_clean
    df_clean, numeric_cols = load_and_clean("data/raw/sold_power_wide_2020_2023.csv")
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_clean(
    raw_path: str,
    missing_threshold: float = 50.0,
    save_path: str = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load raw sold-power CSV and clean it.

    Steps:
        1. Parse timestamps
        2. Drop columns with > missing_threshold % NaN
        3. Forward/backward fill remaining NaN
        4. Add time columns (hour, dow, slot)

    Args:
        raw_path: Path to raw CSV
        missing_threshold: Drop columns with more than this % missing
        save_path: If provided, save cleaned CSV here

    Returns:
        (df_clean, numeric_cols) — cleaned DataFrame and list of company columns
    """
    print(f"Loading {raw_path}...")
    df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    print(f"  Raw shape: {df.shape}")

    # --- Drop high-missing columns ---
    miss_pct = df.isnull().sum() / len(df) * 100
    drop_cols = [
        c for c in miss_pct[miss_pct > missing_threshold].index if c != "timestamp"
    ]
    print(f"  Dropping {len(drop_cols)} columns with >{missing_threshold}% missing")
    df_clean = df.drop(columns=drop_cols).copy()

    # --- Identify numeric columns ---
    numeric_cols = [c for c in df_clean.columns if c != "timestamp"]

    # --- Impute ---
    df_clean[numeric_cols] = df_clean[numeric_cols].ffill().bfill()
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(0)

    # --- Time columns ---
    df_clean["hour"] = df_clean["timestamp"].dt.hour
    df_clean["dow"] = df_clean["timestamp"].dt.dayofweek
    df_clean["slot"] = df_clean["dow"] * 24 + df_clean["hour"]

    print(f"  Clean shape: {df_clean.shape}")
    print(f"  Company columns: {len(numeric_cols)}")
    print(f"  Remaining NaN: {df_clean[numeric_cols].isnull().sum().sum()}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(save_path, index=False)
        print(f"  Saved to {save_path}")

    return df_clean, numeric_cols
