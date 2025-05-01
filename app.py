import streamlit as st
import pandas as pd

st.set_page_config(page_title="AAII Sentiment Risk App", layout="wide")

st.title("📊 AAII Sentiment & S&P 500 Dashboard")

# Load Excel raw
@st.cache_data
def load_raw_excel():
    return pd.read_excel("sentiment_data.xlsx", header=None)

raw_df = load_raw_excel()

# --- Tabs ---
tab1, tab2 = st.tabs(["🗂 Raw Excel Viewer", "📈 Coming Soon: Cleaned & Modeled"])

# --- Tab 1: Raw File Viewer ---
with tab1:
    st.header("🗂 Raw AAII Sentiment Excel File")
    st.dataframe(raw_df)

# --- Tab 2: Placeholder for next phase ---
with tab2:
    st.header("📈 Cleaned Data and Modeling Coming Soon")
    st.markdown("This tab will include charts, cleaned data, and a regression model.")
