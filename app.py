import streamlit as st
import pandas as pd

st.set_page_config(page_title="AAII Sentiment Viewer", layout="wide")

st.title("📄 Raw AAII Sentiment Excel File Viewer")

# Load the Excel file as-is
@st.cache_data
def load_raw_excel():
    return pd.read_excel("sentiment_data.xlsx", header=None)

df = load_raw_excel()

# Show the entire table
st.dataframe(df)
