import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler

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
tab1, tab2, tab3 = st.tabs([
    "🗂 Raw Excel Viewer",
    "📈 Interactive Dashboard",
    "📊 Normalized Comparison"
])

# --- Tab 1: Raw File Viewer ---
with tab1:
    st.header("🗂 Raw AAII Sentiment Excel File")
    st.dataframe(raw_df)

# --- Tab 2: S&P + Sentiment Charts ---
with tab2:
    st.header("📆 Select Time Range for Analysis")

    min_date = clean_df["Date"].min().date()
    max_date = clean_df["Date"].max().date()

    start_date, end_date = st.slider(
        "Select a date range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_df = clean_df[(clean_df["Date"] >= start_date) & (clean_df["Date"] <= end_date)]

    st.subheader("📊 Market Overview")

    # --- Compact Chart 1: S&P 500 ---
    st.markdown("##### 📉 S&P 500 Weekly Close (Log Scale)")
    fig1, ax1 = plt.subplots(figsize=(10, 2.5))
    ax1.plot(filtered_df["Date"], filtered_df["SP500_Close"], color="black")
    ax1.set_yscale("log")
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle="--", linewidth=0.5)
    st.pyplot(fig1)

    # --- Compact Chart 2: Sentiment ---
    st.markdown("##### 🧠 Investor Sentiment")
    fig2, ax2 = plt.subplots(figsize=(10, 2.5))
    ax2.plot(filtered_df["Date"], filtered_df["Bullish"], label="Bullish", color="green")
    ax2.plot(filtered_df["Date"], filtered_df["Neutral"], label="Neutral", color="gray")
    ax2.plot(filtered_df["Date"], filtered_df["Bearish"], label="Bearish", color="red")
    ax2.set_ylabel("Sentiment (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", linewidth=0.5)
    st.pyplot(fig2)

    # --- Data Table ---
    st.subheader("📋 Filtered Data Table")
    st.dataframe(filtered_df, use_container_width=True, height=400)

# --- Tab 3: S&P 500 vs Bullish Sentiment (1-Year MA Option) ---
with tab3:
    st.header("📊 S&P 500 vs. Smoothed Bullish Sentiment (%)")

    # User slider for MA window (1 to 52 weeks)
    ma_window = st.slider("Select Moving Average Window (weeks):", min_value=1, max_value=52, value=4)

    # Apply MA
    df_ma = filtered_df.copy()
    df_ma["Bullish_MA"] = df_ma["Bullish"].rolling(window=ma_window, min_periods=1).mean()

    # Plot
    fig3, ax1 = plt.subplots(figsize=(10, 3))

    # Left y-axis: S&P 500
    ax1.plot(df_ma["Date"], df_ma["SP500_Close"], color="black", label="S&P 500")
    ax1.set_ylabel("S&P 500 Price", color="black")
    ax1.tick_params(axis='y', labelcolor="black")

    # Right y-axis: Bullish sentiment MA
    ax2 = ax1.twinx()
    ax2.plot(df_ma["Date"], df_ma["Bullish_MA"], color="green", label=f"Bullish Sentiment ({ma_window}-Week MA)")
    ax2.set_ylabel("Bullish Sentiment (%)", color="green")
    ax2.tick_params(axis='y', labelcolor="green")

    # Title and grid
    fig3.suptitle(f"S&P 500 vs. Bullish Sentiment ({ma_window}-Week MA)", fontsize=14)
    ax1.grid(True, linestyle="--", linewidth=0.5)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    st.pyplot(fig3)
