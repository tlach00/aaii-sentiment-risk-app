import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime

st.set_page_config(page_title="AAII Sentiment Risk App", layout="wide")

st.title("📊 AAII Sentiment & S&P 500 Dashboard")

# --- Load raw Excel ---
@st.cache_data
def load_raw_excel():
    return pd.read_excel("sentiment_data.xlsx", header=None)

# --- Load cleaned data ---
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
tab1, tab2 = st.tabs(["🗂 Raw Excel Viewer", "📈 Interactive Dashboard"])

# --- Tab 1: Raw File Viewer ---
with tab1:
    st.header("🗂 Raw AAII Sentiment Excel File")
    st.dataframe(raw_df)

# --- Tab 2: Charts + Interactive Table ---
with tab2:
    st.header("📆 Select Time Range for Analysis")

    # Set up proper date range
    min_date = clean_df["Date"].min().date()
    max_date = clean_df["Date"].max().date()

    # Slider input
    start_date, end_date = st.slider(
        "Select a date range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    # Convert slider output back to Timestamp
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter data
    filtered_df = clean_df[(clean_df["Date"] >= start_date) & (clean_df["Date"] <= end_date)]

    # --- Chart 1: S&P 500 ---
    st.subheader("📉 S&P 500 Weekly Close (Log Scale)")
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(filtered_df["Date"], filtered_df["SP500_Close"], color="black")
    ax1.set_yscale("log")
    ax1.set_ylabel("Price")
    ax1.set_title("S&P 500 Weekly Close", fontsize=14)
    ax1.grid(True, linestyle="--", linewidth=0.5)
    st.pyplot(fig1)

    # --- Chart 2: Sentiment ---
    st.subheader("🧠 AAII Sentiment Over Time")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(filtered_df["Date"], filtered_df["Bullish"], label="Bullish", color="green")
    ax2.plot(filtered_df["Date"], filtered_df["Neutral"], label="Neutral", color="gray")
    ax2.plot(filtered_df["Date"], filtered_df["Bearish"], label="Bearish", color="red")
    ax2.set_ylabel("Sentiment (%)")
    ax2.set_title("Investor Sentiment", fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle="--", linewidth=0.5)
    st.pyplot(fig2)

    # --- Table ---
    st.subheader("📋 Filtered Data Table")
    st.dataframe(filtered_df, use_container_width=True, height=400)
