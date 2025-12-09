# Predictive Anomaly Detection for Colombia Power Grid  
### End-to-end data pipeline: Transform → Normalize (2020–2023)

This repository contains the data engineering pipeline used to prepare Colombia’s hourly power-sold dataset for anomaly detection using an LSTM Autoencoder and EXAMM-evolved RNNs. The pipeline has two main stages:

1. Data Transformation – Convert heterogeneous Excel files into clean, continuous hourly matrices.  
2. Data Normalization – Apply leak-free Z-score normalization using only 2020–2022 statistics.

---

## 📁 Repository Structure

colombia-grid-forecasting/  
│  
├─ src/  
│  ├─ build_all_years.py                  (Transform raw Excel → wide hourly CSV)  
│  └─ normalize_power.py                  (Normalize dataset with leak-free Z-scores)  
│  
├─ datasets/  
│  ├─ raw/                                (Raw SEME Excel files 2020–2023)  
│  ├─ transformed/                        (Wide dataset 2020–2023)  
│  │   └─ sold_power_wide_2020_2023.csv  
│  └─ normalized/                         (Normalization outputs)  
│      ├─ sold_power_wide_2020_2023_normalized.csv  
│      └─ sold_power_wide_normalization_params.csv  
│  
└─ docs/                                  (Flowcharts, diagrams, notes)

---

## 🔧 Environment Setup

python -m venv .venv  
source .venv/bin/activate      (Windows: .venv\Scripts\activate)  
pip install pandas numpy openpyxl pyarrow  

---

## 🅰️ STEP 1 — Data Transformation (Excel → hourly wide CSVs)

The script build_all_years.py processes the raw SEME Excel files, which contain:

• Fecha (date)  
• Codigo Comercializador (provider ID)  
• Mercado (market segment)  
• Hour columns 0–23  

What the script does:

1. Detects header row  
2. Converts 24 hour columns into long format  
3. Builds hourly timestamps  
4. Creates seller_market = Codigo Comercializador + Mercado  
5. Pivots into wide format (one column per provider)  
6. Reindexes to complete hourly timeline (leap-year safe)  
7. Forward-fills only short gaps (≤ 2 hours)  
8. Saves per-year CSVs + combined 2020–2023 CSV  

Run:

cd src  
python build_all_years.py  

Outputs (datasets/transformed/):

sold_power_wide_2020.csv  
sold_power_wide_2021.csv  
sold_power_wide_2022.csv  
sold_power_wide_2023.csv  
sold_power_wide_2020_2023.csv  

---

## 🅱️ STEP 2 — Data Normalization (Leak-Free Z-Score)

The script normalize_power.py prepares the dataset for EXAMM + LSTM models.

Key idea:  
Use only 2020–2022 to compute mean/std → prevents leakage into 2023.

What the script does:

1. Loads sold_power_wide_2020_2023.csv  
2. Identifies provider numeric columns  
3. Extracts 2020–2022 rows for training statistics  
4. Computes mean (μ) and std (σ) for each provider  
5. Applies Z-score normalization (x−μ)/σ  
6. Sets zero-variance columns to 0  
7. Saves normalized dataset + params CSV  

Run:

cd src  
python normalize_power.py  

Outputs (datasets/normalized/):

sold_power_wide_2020_2023_normalized.csv  
sold_power_wide_normalization_params.csv  

---

## 🧪 Reproducibility Notes

• Pipeline is deterministic  
• Hourly timestamps aligned across all years  
• Only short gaps (≤ 2 hours) are imputed  
• No leakage: normalization uses 2020–2022 stats  
• Parameter CSV ensures reproducibility  

---

## 👥 Authors

Aparajita Pavan  
Tharak Bhupathi  
Karthik Pachabatla  
Advisor: Dr. Travis Desell — Rochester Institute of Technology  

---

## 📄 License

-
