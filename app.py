import streamlit as st
import pandas as pd
from io import BytesIO
from process_csv import process_csv  # This is your notebook logic as a function

st.title("CSV Blanks Filler App")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Original CSV", df)

    # Process CSV
    processed_df = process_csv(df)
    
    st.write("### Processed CSV", processed_df)

    # Convert to CSV and download
    def convert_df_to_csv(df):
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return output

    st.download_button(
        label="Download Processed CSV",
        data=convert_df_to_csv(processed_df),
        file_name="processed.csv",
        mime="text/csv",
    )
