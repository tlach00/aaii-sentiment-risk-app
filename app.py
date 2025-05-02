import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime

st.set_page_config(
    page_title="Sentiment Strategy Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

st.title(":chart_with_upwards_trend: Sentiment vs S&P 500 Strategy")

@st.cache_data

def load_data():
    df = pd.read_excel("sentiment_data.xlsx", skiprows=7, usecols="A:D,M", header=None)
    df.columns = ["Date", "Bullish", "Neutral", "Bearish", "SP500_Close"]
    df = df.dropna()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.set_index("Date").resample("D").ffill().dropna().copy()
    df["SP500_Return"] = df["SP500_Close"].pct_change()
    df["Sentiment"] = df["Bullish"]
    return df.dropna()

df = load_data()

st.sidebar.header("Strategy Parameters")
rolling_window = st.sidebar.slider("Rolling Window (days)", 5, 60, 20, step=5)
initial_capital = st.sidebar.number_input("Initial Capital ($)", 1000, 1000000, 10000, step=1000)

# Compute rolling z-scores
df["Z_Sentiment"] = (df["Sentiment"] - df["Sentiment"].rolling(window=rolling_window).mean()) / df["Sentiment"].rolling(window=rolling_window).std()
df["Z_Price"] = (df["SP500_Close"] - df["SP500_Close"].rolling(window=rolling_window).mean()) / df["SP500_Close"].rolling(window=rolling_window).std()

# Generate signals
df["Signal"] = 0
df.loc[df["Z_Sentiment"] > df["Z_Price"], "Signal"] = 1
df.loc[df["Z_Sentiment"] < df["Z_Price"], "Signal"] = -1
df["Position"] = df["Signal"].shift(1).fillna(0)

# Returns
df["Strategy_Return"] = df["SP500_Return"] * df["Position"]
df = df.dropna()
df["BuyHold"] = initial_capital * (1 + df["SP500_Return"]).cumprod()
df["Strategy"] = initial_capital * (1 + df["Strategy_Return"]).cumprod()

# Time range slider
from_date, to_date = st.slider(
    "Select date range",
    min_value=df.index.min().date(),
    max_value=df.index.max().date(),
    value=(df.index.min().date(), df.index.max().date()),
    format="YYYY-MM-DD"
)
df_filtered = df[(df.index.date >= from_date) & (df.index.date <= to_date)]

st.header("Portfolio Growth", divider="gray")
st.line_chart(df_filtered[["Strategy", "BuyHold"]])

# Metrics block
st.header(f"Performance Summary as of {to_date}", divider="gray")
latest = df_filtered.iloc[-1]
start = df_filtered.iloc[0]

strat_return = latest["Strategy"] / start["Strategy"] - 1
bh_return = latest["BuyHold"] / start["BuyHold"] - 1

cols = st.columns(2)

with cols[0]:
    st.metric("Strategy Return", f"{latest['Strategy']:,.0f} $", delta=f"{strat_return:.2%}")

with cols[1]:
    st.metric("Buy & Hold Return", f"{latest['BuyHold']:,.0f} $", delta=f"{bh_return:.2%}")

st.header("Z-Score Inputs", divider="gray")
st.line_chart(df_filtered[["Z_Sentiment", "Z_Price"]])

st.header("Signal Positioning", divider="gray")
st.line_chart(df_filtered[["Position"]])
