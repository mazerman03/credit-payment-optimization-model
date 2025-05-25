import pandas as pd

# Load the CSV
input_file = 'data/output/clean_tries.csv'  # Replace with your actual file name
df = pd.read_csv(input_file)

# Convert 'fechaEnvioCobro' to datetime
df['fechaEnvioCobro'] = pd.to_datetime(df['fechaEnvioCobro'], format='%d/%m/%Y', errors='coerce')

# Filter for rows in 2025
df_2025 = df[df['fechaEnvioCobro'].dt.year == 2025]

# Keep only the desired columns
columns_to_keep = ['idCredito', 'montoExigible', 'fechaEnvioCobro', 'idBanco']
df_2025 = df_2025[columns_to_keep]

# Save to CSV
df_2025.to_csv('test2025.csv', index=False)

print("✅ Output with selected columns saved to test2025.csv")
