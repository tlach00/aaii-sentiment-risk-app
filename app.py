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

# --- Tab 3: Normalized Overlay Plot ---
with tab3:
    st.header("📊 Normalized S&P 500 and Bullish Sentiment")

    df_norm = filtered_df.copy()
    scaler = MinMaxScaler()
    df_norm[["Norm_SP500", "Norm_Bullish"]] = scaler.fit_transform(
        df_norm[["SP500_Close", "Bullish"]]
    )

    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(df_norm["Date"], df_norm["Norm_SP500"], label="S&P 500", color="black")
    ax3.plot(df_norm["Date"], df_norm["Norm_Bullish"], label="Bullish Sentiment", color="green")
    ax3.set_ylabel("Normalized (0–1)")
    ax3.set_title("Normalized Price vs Sentiment", fontsize=14)
    ax3.legend()
    ax3.grid(True, linestyle="--", linewidth=0.5)
    st.pyplot(fig3)
    

