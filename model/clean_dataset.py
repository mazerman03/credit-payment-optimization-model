import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV (replace with your actual file path)
df = pd.read_csv("data/completo.csv", dtype=str)

# Parse date using Mexican format (day/month/year)
df["fechaCreacionLista"] = pd.to_datetime(df["fechaCreacionLista"], format="%d/%m/%Y %I:%M%p", errors="coerce")

# Extract just the date part
df["fecha_dia"] = df["fechaCreacionLista"].dt.date

# Flag successful transactions (code 0 means success)
df["is_success"] = df["idRespuestaBanco"] == "0"

# Group by credit and day
def filter_day_group(group):
    if group["is_success"].any():
        # If there's a success that day, keep only the successful ones
        return group[group["is_success"]]
    else:
        # If no success, keep just the first failure
        return group.head(1)

# Apply the logic per credit per date
filtered = df.groupby(["idCredito", "fecha_dia"], group_keys=False).apply(filter_day_group)

# Drop helper columns
filtered = filtered.drop(columns=["is_success", "fecha_dia"])

# Save cleaned dataset
#filtered.to_csv("data/output/clean_tries.csv", index=False)



grouped = filtered.copy()
grouped["montoCobrado"] = pd.to_numeric(grouped["montoCobrado"], errors="coerce").fillna(0)
grouped["montoCobrar"] = pd.to_numeric(grouped["montoCobrar"], errors="coerce").fillna(0)
grouped["is_success"] = grouped["idRespuestaBanco"] == "0"

metrics = grouped.groupby("idCredito").agg(
    total_attempts=("is_success", "count"),
    total_successes=("is_success", "sum"),
    total_failures=("is_success", lambda x: (~x).sum()),
    total_amount_paid=("montoCobrado", "sum"),
    total_amount_due=("montoCobrar", "sum")
)

metrics["success_rate"] = metrics["total_successes"] / metrics["total_attempts"]
metrics["repayment_ratio"] = metrics["total_amount_paid"] / metrics["total_amount_due"]

metrics_sorted = metrics.sort_values(by="repayment_ratio", ascending=False)
#metrics_sorted.to_csv("credit_repayment_metrics.csv")


# Load the metrics (or use `metrics_sorted` from earlier)
metrics = pd.read_csv("data/output/credit_repayment_metrics.csv")

# Set plot style
sns.set_theme(style="whitegrid")

# Plot histogram or KDE
plt.figure(figsize=(10, 6))
sns.histplot(metrics["repayment_ratio"], bins=20, kde=True, color='steelblue')

# Labels and title
plt.title("Distribution of Repayment Ratios by Credit", fontsize=16)
plt.xlabel("Repayment Ratio", fontsize=12)
plt.ylabel("Number of Credits", fontsize=12)

# Show plot
plt.tight_layout()
plt.show()