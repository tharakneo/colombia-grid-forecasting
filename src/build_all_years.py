# build_all_years.py — process 2020–2023 from one folder
import sys
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).parent
SRC = HERE  # Excel files are here
OUT = HERE / "out_all"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = [2020, 2021, 2022, 2023]
HOUR_COLS = [str(h) for h in range(24)]
REQ_COLS = {"Fecha", "Codigo Comercializador", "Mercado"}
IMPUTE_SHORT_GAP_HOURS = 2  # set to 0 to disable short gap filling


def log(msg):
    print(msg, flush=True)


def find_header_row(df: pd.DataFrame) -> int:
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).str.strip().tolist()
        if "Fecha" in row and "Codigo Comercializador" in row:
            return i
    raise ValueError("Header row not found (need 'Fecha' & 'Codigo Comercializador').")


def read_one(fp: Path) -> pd.DataFrame:
    log(f"  - reading: {fp.name}")
    raw = pd.read_excel(fp, sheet_name=0, header=None, engine="openpyxl")
    hdr = find_header_row(raw)
    df = pd.read_excel(fp, sheet_name=0, header=hdr, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    missing = REQ_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{fp.name} missing columns: {missing}")

    hours_present = [c for c in HOUR_COLS if c in df.columns]
    if not hours_present:
        raise ValueError(f"{fp.name} has no hour columns 0..23")

    m = df.melt(
        id_vars=["Fecha", "Codigo Comercializador", "Mercado"],
        value_vars=hours_present,
        var_name="hour",
        value_name="demanda_mwh",
    )
    m["hour"] = pd.to_numeric(m["hour"], errors="coerce").astype("Int64")
    m["demanda_mwh"] = pd.to_numeric(m["demanda_mwh"], errors="coerce")
    m["timestamp"] = pd.to_datetime(m["Fecha"], errors="coerce") + pd.to_timedelta(
        m["hour"].fillna(0).astype(float), unit="h"
    )
    m["seller_market"] = (
        m["Codigo Comercializador"].astype(str).str.strip().str.upper()
        + " "
        + m["Mercado"].astype(str).str.strip().str.upper()
    )
    m = m.dropna(subset=["timestamp"])
    return m[["timestamp", "seller_market", "demanda_mwh"]]


def short_gap_fill(wide: pd.DataFrame, max_h: int) -> pd.DataFrame:
    if max_h <= 0:
        return wide
    for c in wide.columns:
        s = wide[c]
        mask = s.isna()
        if mask.any():
            grp = (mask != mask.shift()).cumsum()
            run_len = mask.groupby(grp).transform("sum")
            to_ffill = mask & (run_len <= max_h)
            s.loc[to_ffill] = s.ffill().loc[to_ffill]
            wide[c] = s
    return wide


def pivot_and_save(long_all: pd.DataFrame):
    # Per-year processing
    for y in YEARS:
        log(f"\n=== YEAR {y} ===")
        yr_long = long_all[(long_all["timestamp"].dt.year == y)].copy()
        if yr_long.empty:
            log(f"  (no rows for {y}, skipping)")
            continue

        wide = yr_long.pivot_table(
            index="timestamp",
            columns="seller_market",
            values="demanda_mwh",
            aggfunc="sum",
        ).sort_index()

        # full hourly index (leap year handled automatically)
        start = pd.Timestamp(f"{y}-01-01 00:00:00")
        end = pd.Timestamp(f"{y}-12-31 23:00:00")
        full_idx = pd.date_range(start, end, freq="H")
        wide = wide.reindex(full_idx)

        wide = short_gap_fill(wide, IMPUTE_SHORT_GAP_HOURS).astype(float)

        out_csv = OUT / f"sold_power_wide_{y}.csv"
        wide.to_csv(out_csv, index_label="timestamp", date_format="%Y-%m-%d %H:%M:%S")
        log(f"  SAVED: {out_csv}  shape={wide.shape}")

    # Optional combined (continuous across all years present)
    if not long_all.empty:
        all_wide = long_all.pivot_table(
            index="timestamp",
            columns="seller_market",
            values="demanda_mwh",
            aggfunc="sum",
        ).sort_index()
        full_idx = pd.date_range(all_wide.index.min(), all_wide.index.max(), freq="H")
        all_wide = short_gap_fill(
            all_wide.reindex(full_idx), IMPUTE_SHORT_GAP_HOURS
        ).astype(float)
        out_all = OUT / "sold_power_wide_2020_2023.csv"
        all_wide.to_csv(
            out_all, index_label="timestamp", date_format="%Y-%m-%d %H:%M:%S"
        )
        log(f"\nSAVED combined: {out_all}  shape={all_wide.shape}")


def main():
    log(f"Folder: {SRC}")
    files = sorted(SRC.glob("Demanda_Comercial_Por_Comercializador_SEME*.xlsx"))
    if not files:
        files = sorted(SRC.glob("*.xlsx"))
    if not files:
        sys.exit("No Excel files found in this folder.")

    log(f"Found {len(files)} Excel files:")
    for f in files:
        log(f"  • {f.name}")

    parts = [read_one(f) for f in files]
    long_all = pd.concat(parts, ignore_index=True)
    log(f"\nTotal long rows: {len(long_all):,}")
    pivot_and_save(long_all)


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    main()
