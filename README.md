# Predictive Anomaly Detection — Colombia Power Grid

**Capstone Project | RIT MS Data Science**  
Authors: Aparajita Pavan, Tharak Bhupathi, Karthik Pachabatla  
Advisor: Dr. Travis Desell

## Project Structure

```
colombia_power_grid_anomaly/
├── run_pipeline.py              # ← Run this! Main entry point
├── requirements.txt             # Python dependencies
├── configs/
│   └── config.yaml              # All parameters in one place
├── data/
│   ├── raw/                     # Put sold_power_wide_2020_2023.csv here
│   └── processed/               # Cleaned data + features (auto-generated)
├── src/
│   ├── preprocessing/
│   │   └── clean.py             # Load, drop high-missing cols, impute
│   ├── features/
│   │   └── slot_features.py     # Slot-calibrated z-scores + 9 features
│   ├── models/
│   │   ├── detectors.py         # Isolation Forest + One-Class SVM
│   │   └── reasons.py           # Reason labels + top contributing companies
│   ├── explainability/
│   │   └── explain.py           # SHAP + LIME explanations
│   └── visualization/
│       └── dashboard.py         # 8-panel dashboard + deep-dive plots
├── notebooks/                   # Jupyter notebooks for exploration
├── outputs/
│   ├── figures/                 # All plots (auto-generated)
│   ├── reports/                 # anomaly_report.csv
│   └── scores/                  # Full scored dataset
└── tests/                       # Unit tests
```

## Quick Start

### 1. Setup
```bash
# Clone or download this project
cd colombia_power_grid_anomaly

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Data
Place `sold_power_wide_2020_2023.csv` in `data/raw/`:
```bash
cp /path/to/sold_power_wide_2020_2023.csv data/raw/
```

### 3. Run
```bash
# Full pipeline (with SHAP + LIME — takes ~5 min)
python run_pipeline.py

# Quick run (skip explainability — takes ~1 min)
python run_pipeline.py --skip-shap --skip-lime

# Custom data path
python run_pipeline.py --data /path/to/your_data.csv
```

### 4. Check Outputs
- `outputs/figures/anomaly_dashboard.png` — 8-panel dashboard
- `outputs/figures/anomaly_deepdive.png` — Hour-by-hour top anomaly days
- `outputs/figures/shap_summary.png` — SHAP global importance
- `outputs/figures/shap_waterfall_*.png` — Per-anomaly SHAP
- `outputs/figures/lime_anomaly_*.png` — Per-anomaly LIME
- `outputs/reports/anomaly_report.csv` — All anomalies with reasons + companies
- `outputs/scores/anomaly_features_full.csv` — Full dataset with scores

## Pipeline Overview

1. **Preprocessing**: Drop columns >50% missing, forward/backward fill NaN
2. **Slot-Calibrated Features**: Compare each reading to its hour-of-week median using IQR-based z-scores (168 slots = 24h × 7 days)
3. **9 Aggregate Features**: mean_abs_slot_z, max_slot_z, n_flagged (3z/5z), total_power_slot_z, delta_total, rolling_vol_6h, HHI, top5_share
4. **Anomaly Detection**: Isolation Forest + One-Class SVM → consensus (both must agree)
5. **Reason Labels**: SYSTEM_DROP, SYSTEM_SPIKE, WIDESPREAD, EXTREME_SINGLE_COMPANY, MARKET_CONCENTRATION, SUDDEN_CHANGE, HIGH_VOLATILITY
6. **Explainability**: SHAP (global + per-anomaly waterfall) + LIME (local feature contributions)

## Key Findings

- ~146 consensus anomalies (0.27% of 35,064 hours)
- Dominant pattern: **SYSTEM_DROP** on holidays (Jan 1, Dec 25) and April 2020 (COVID)
- Most important feature: `mean_abs_slot_z` (average "weirdness" across companies)
- Top contributors are identifiable per anomaly with z-scores and raw values

## Next Steps

- [ ] Integrate EXAMM-evolved RNN for temporal residual features
- [ ] Add COPOD as third detector baseline
- [ ] Online calibration / adaptive thresholding
- [ ] Bidding curve anomaly detection
