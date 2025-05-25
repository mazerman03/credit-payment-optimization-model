import pandas as pd

def process_csv(df: pd.DataFrame) -> pd.DataFrame:
    # Example: fill blanks (NaNs) with forward fill
    return df.fillna(method='ffill')