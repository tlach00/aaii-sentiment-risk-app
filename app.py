# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_excel("sentiment_data.xlsx", skiprows=4)
    df = df.rename(columns={df.columns[0]: "Date"})
    df = df[df["Date"].notnull()]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

df = load_data()

# ---------- Streamlit App ----------
st.set_page_config(page_title="AAII Sentiment Dashboard", layout="wide")

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["📄 Raw Excel Viewer", "📊 Interactive Dashboard"])

# ---------- Tab 1 ----------
with tab1:
    st.header("📄 Raw AAII Sentiment Excel File")
    st.dataframe(df, use_container_width=True)

# ---------- Tab 2 ----------
with tab2:
    st.header("📊 S&P 500 Weekly Close (Log Scale)")
    
    min_date = df["Date"].min()
    max_date = df["Date"].max()

    start_date, end_date = st.slider(
        "Select Time Range for Analysis",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(filtered_df["Date"], filtered_df["S&P 500 Weekly Close"], color="black")
    ax1.set_yscale("log")
    ax1.set_ylabel("S&P 500 Price")
    ax1.set_title("S&P 500 Weekly Close")
    ax1.grid(True)

    st.pyplot(fig)

    st.header("📉 AAII Sentiment Over Time")

    st.markdown("**Select sentiment components to display:**")
    show_bull = st.checkbox("🐂 Bullish", True)
    show_neut = st.checkbox("⚪ Neutral", True)
    show_bear = st.checkbox("🐻 Bearish", True)

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    if show_bull:
        ax2.plot(filtered_df["Date"], filtered_df["Bullish"], label="Bullish", color="green")
    if show_neut:
        ax2.plot(filtered_df["Date"], filtered_df["Neutral"], label="Neutral", color="gray")
    if show_bear:
        ax2.plot(filtered_df["Date"], filtered_df["Bearish"], label="Bearish", color="red")

    ax2.set_ylabel("Sentiment (%)")
    ax2.set_title("Investor Sentiment")
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)
