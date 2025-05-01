import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="AAII Sentiment Risk App", layout="wide")

st.title("\U0001F4CA AAII Sentiment & S&P 500 Dashboard")

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
    "\U0001F5C2 Raw Excel Viewer",
    "\U0001F4C8 Interactive Dashboard",
    "\U0001F4CA Smoothed Sentiment vs S&P"
])

# --- Tab 1: Raw File Viewer ---
with tab1:
    st.header("\U0001F5C2 Raw AAII Sentiment Excel File")
    st.dataframe(raw_df)

# --- Tab 2: Interactive Charts + MA + Checkboxes ---
with tab2:
    st.header("\U0001F4C6 Select Time Range for Analysis")

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

    st.markdown("##### \U0001F4C9 S&P 500 Weekly Close (Log Scale)")
    fig1, ax1 = plt.subplots(figsize=(10, 2))
    ax1.plot(filtered_df["Date"], filtered_df["SP500_Close"], color="black")
    ax1.set_yscale("log")
    ax1.set_ylabel("Price", fontsize=8)
    ax1.tick_params(axis='both', labelsize=8)
    ax1.grid(True, linestyle="--", linewidth=0.3)
    st.pyplot(fig1)

    st.markdown("##### \U0001F9E0 Investor Sentiment (Toggle Lines)")
    col1, col2, col3 = st.columns(3)
    show_bullish = col1.checkbox("\U0001F402 Bullish", value=True)
    show_neutral = col2.checkbox("☰ Neutral", value=True)
    show_bearish = col3.checkbox("\U0001F43B Bearish", value=True)

    fig2, ax2 = plt.subplots(figsize=(10, 2))
    if show_bullish:
        ax2.plot(filtered_df["Date"], filtered_df["Bullish"], label="Bullish", color="green")
    if show_neutral:
        ax2.plot(filtered_df["Date"], filtered_df["Neutral"], label="Neutral", color="gray")
    if show_bearish:
        ax2.plot(filtered_df["Date"], filtered_df["Bearish"], label="Bearish", color="red")
    ax2.set_ylabel("Sentiment (%)", fontsize=8)
    ax2.tick_params(axis='both', labelsize=8)
    ax2.legend()
    ax2.grid(True, linestyle="--", linewidth=0.3)
    st.pyplot(fig2)

    st.markdown("##### \U0001F4C8 Bullish Sentiment Moving Average")
    ma_window = st.slider("Select MA Window (weeks):", 1, 52, 4, key="tab2_ma")

    df_ma = filtered_df.copy()
    df_ma["Bullish_MA"] = df_ma["Bullish"].rolling(window=ma_window, min_periods=1).mean()

    fig3, ax3 = plt.subplots(figsize=(10, 2))
    ax3.plot(df_ma["Date"], df_ma["SP500_Close"], color="black", label="S&P 500")
    ax4 = ax3.twinx()
    ax4.plot(df_ma["Date"], df_ma["Bullish_MA"], color="green", label=f"Bullish ({ma_window}-W MA)")
    ax3.set_ylabel("S&P 500 Price", fontsize=8, color="black")
    ax4.set_ylabel("Bullish Sentiment (%)", fontsize=8, color="green")
    ax3.tick_params(axis='both', labelsize=8, labelcolor="black")
    ax4.tick_params(axis='both', labelsize=8, labelcolor="green")
    ax3.grid(True, linestyle="--", linewidth=0.3)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    st.pyplot(fig3)

    st.subheader("\U0001F4CB Filtered Data Table")
    st.dataframe(filtered_df, use_container_width=True, height=400)

# --- Tab 3: Dual Y-Axis — S&P 500 vs Smoothed Sentiment ---
with tab3:
    st.header("\U0001F4CA S&P 500 vs. Smoothed Bullish Sentiment (%)")

    ma_window = st.slider("Select Moving Average Window (weeks):", 1, 52, 4, key="tab3_ma")

    df_ma = filtered_df.copy()
    df_ma["Bullish_MA"] = df_ma["Bullish"].rolling(window=ma_window, min_periods=1).mean()

    fig3, ax1 = plt.subplots(figsize=(10, 2))
    ax1.plot(df_ma["Date"], df_ma["SP500_Close"], color="black", label="S&P 500")
    ax1.set_ylabel("S&P 500 Price", fontsize=8, color="black")
    ax1.tick_params(axis='y', labelcolor="black", labelsize=8)

    ax2 = ax1.twinx()
    ax2.plot(df_ma["Date"], df_ma["Bullish_MA"], color="green", label=f"\U0001F402 Bullish ({ma_window}-W MA)")
    ax2.set_ylabel("Bullish Sentiment (%)", fontsize=8, color="green")
    ax2.tick_params(axis='y', labelcolor="green", labelsize=8)

    ax1.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='x', labelsize=8)
    ax1.grid(True, linestyle="--", linewidth=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    st.pyplot(fig3)
