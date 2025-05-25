import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "test2025.csv"
METRICS_FILE = "data/output/credit_repayment_metrics.csv"
STRATEGY_FILE = "data/raw/CatEmisora_Merged.csv"
OUTPUT_FILE = "repayment_simulation_test_1.csv"

# Load input + metrics
input_df = pd.read_csv(INPUT_FILE)
input_df["idCredito"] = input_df["idCredito"].astype(str)

metrics_df = pd.read_csv(METRICS_FILE)
metrics_df["idCredito"] = metrics_df["idCredito"].astype(str)

# Merge repayment ratio
merged = pd.merge(input_df, metrics_df[["idCredito", "repayment_ratio"]], on="idCredito", how="left")

# Expected repayment
merged["expected_repayment"] = merged["montoExigible"] * merged["repayment_ratio"]

# Seasonality score
def seasonality_score(day):
    radians = (day / 30) * 2 * np.pi * 3
    return (np.cos(radians) + 1) / 2

merged["fechaEnvioCobro"] = pd.to_datetime(merged["fechaEnvioCobro"], errors='coerce')
merged["day_of_month"] = merged["fechaEnvioCobro"].dt.day
merged["seasonality_score"] = merged["day_of_month"].apply(lambda d: seasonality_score(d) if not pd.isna(d) else np.nan)

# Adjusted repayment
merged["adjusted_expected_repayment"] = merged["expected_repayment"] * merged["seasonality_score"]

# Risk classification
def classify_risk(ratio):
    if pd.isna(ratio):
        return "unknown"
    return "highrisk" if ratio < 0.5 else "lowrisk"

merged["risk_level"] = merged["repayment_ratio"].apply(classify_risk)

# Load strategy catalog
strategy_df = pd.read_csv(STRATEGY_FILE)

# Strategy selection logic
def select_strategy(row, strategy_df):
    if pd.isna(row["idBanco"]):
        return np.nan

    strategies = strategy_df[strategy_df["IdBanco"] == row["idBanco"]]

    if strategies.empty:
        return np.nan

    # Filter by risk
    if row["risk_level"] == "highrisk":
        # Prefer higher risk strategies
        chosen = strategies.sort_values(by="Riesgo", ascending=False).iloc[0]
    else:
        # Prefer lower cost and risk
        chosen = strategies.sort_values(by=["Riesgo", "EmisoraCostoAceptado"]).iloc[0]

    return chosen["idEmisora"]

merged["idEmisora"] = merged.apply(lambda row: select_strategy(row, strategy_df), axis=1)

# Save final result
merged[[
    "idCredito",
    "montoExigible",
    "fechaEnvioCobro",
    "idEmisora",
    "idBanco",
    "seasonality_score",
    "adjusted_expected_repayment"
]].to_csv(OUTPUT_FILE, index=False)

print(f"✅ Final strategy-based results saved to {OUTPUT_FILE}")
