import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AAII Sentiment Risk App", layout="wide")

st.title("📊 AAII Sentiment & S&P 500 Dashboard")

# --- Load Excel (raw full dump) ---
@st.cache_data
def load_raw_excel():
    return pd.read_excel("sentiment_data.xlsx", header=None)

# --- Load cleaned relevant subset ---
@st.cache_data
def load_clean_data():
    df = pd.read_excel("sentiment_data.xlsx", skiprows=7, usecols="A:D,M", header=None)
    df.columns = ["Date", "Bullish", "Neutral", "Bearish", "SP500_Close"]
    df = df.dropna(subset=["Date", "Bullish", "Neutral", "Bearish", "SP500_Close"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    df["SP500_Return"] = df["SP500_Close"].pct_change() * 100
    return df.dropna()

# Load data
raw_df = load_raw_excel()
clean_df = load_clean_data()

# --- Tabs ---
tab1, tab2 = st.tabs(["🗂 Raw Excel Viewer", "📈 Cleaned Data & Price Chart"])

# --- Tab 1: Raw File Viewer ---
with tab1:
    st.header("🗂 Raw AAII Sentiment Excel File")
    st.dataframe(raw_df)

# --- Tab 2: Cleaned Chart and Table ---
with tab2:
    st.header("📉 S&P 500 Weekly Close Price")

    # Smaller chart + log scale
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(clean_df["Date"], clean_df["SP500_Close"], color="black")
    ax.set_yscale("log")
    ax.set_title("S&P 500 Weekly Close (Log Scale)", fontsize=14)
    ax.set_ylabel("Price (log scale)")
    ax.grid(True, linestyle="--", linewidth=0.5)
    st.pyplot(fig)

    st.subheader("🧾 Cleaned Sentiment + S&P 500 Data")
    st.dataframe(clean_df)
