import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime

# Set up the Streamlit page configuration
st.set_page_config(page_title="AAII Sentiment & S&P 500 Dashboard", layout="wide")

# Function to load data
@st.cache_data

def load_data():
    file_path = "sentiment_data.xlsx"
    df = pd.read_excel(file_path, skiprows=4)
    df = df.rename(columns={
        df.columns[0]: "Date",
        df.columns[1]: "Bullish",
        df.columns[2]: "Neutral",
        df.columns[3]: "Bearish",
        df.columns[12]: "SP500_Weekly_Close"
    })
    df = df.dropna(subset=["Date", "SP500_Weekly_Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] >= "1988-01-01"]
    df[["Bullish", "Neutral", "Bearish"]] = df[["Bullish", "Neutral", "Bearish"]].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=["Bullish", "Neutral", "Bearish"])
    df = df.sort_values("Date")
    df["SP500_Return"] = df["SP500_Weekly_Close"].pct_change() * 100
    df["Bullish_MA"] = df["Bullish"].rolling(window=52, min_periods=1).mean()
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.title("📊 AAII Sentiment & S&P 500 Dashboard")
page = st.sidebar.radio("", ["📄 Raw Excel Viewer", "📈 Interactive Dashboard"])

# --- Page 1: Raw Excel Viewer ---
if page == "📄 Raw Excel Viewer":
    st.header("📄 Raw AAII Sentiment Excel File")
    st.dataframe(df, use_container_width=True)

# --- Page 2: Interactive Dashboard ---
elif page == "📈 Interactive Dashboard":
    st.header("📈 Market Sentiment & S&P 500 Risk Model")

    # Time Range Slider
    min_date, max_date = df["Date"].min(), df["Date"].max()
    start_date, end_date = st.slider(
        "📅 Select Time Range for Analysis",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM"
    )
    mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
    df_filtered = df.loc[mask]

    # First chart: S&P 500 log-scale
    st.subheader("📉 S&P 500 Weekly Close (Log Scale)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_filtered["Date"], df_filtered["SP500_Weekly_Close"], color="black")
    ax.set_yscale("log")
    ax.set_title("S&P 500 Weekly Close")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    st.pyplot(fig)

    # Second chart: Sentiment bars with optional smoothing
    st.subheader("🧠 AAII Sentiment Over Time")
    sentiment_options = ["🐂 Bullish", "😐 Neutral", "🐻 Bearish"]
    selected_sentiments = st.multiselect(
        "Select Sentiment Components to Display",
        sentiment_options,
        default=sentiment_options
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {"🐂 Bullish": "green", "😐 Neutral": "gray", "🐻 Bearish": "red"}
    for label in selected_sentiments:
        col = label.split()[1]  # Extract column name from emoji label
        ax.bar(df_filtered["Date"], df_filtered[col], label=label, alpha=0.5, color=colors[label])
    ax.set_title("Investor Sentiment")
    ax.set_ylabel("Sentiment (%)")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    st.pyplot(fig)

    # Moving average chart
    st.subheader("📊 Bullish Sentiment (1-Year Moving Average)")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df_filtered["Date"], df_filtered["Bullish_MA"], label="1-Year MA", color="green")
    ax.set_ylabel("Bullish Sentiment (%)")
    ax.set_title("Smoothed Bullish Sentiment Over Time")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    st.pyplot(fig)
