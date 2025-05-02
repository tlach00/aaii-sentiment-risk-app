# sentiment_backtest.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment Strategy Backtest", layout="wide")

st.title("\U0001F4C8 Sentiment-Based Backtest Strategy")

@st.cache_data

def load_data():
    df = pd.read_excel("sentiment_data.xlsx", skiprows=7, usecols="A:D,M", header=None)
    df.columns = ["Date", "Bullish", "Neutral", "Bearish", "SP500_Close"]
    df.dropna(subset=["Date", "Bullish", "Bearish", "SP500_Close"], inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.sort_values("Date", inplace=True)
    df.dropna(subset=["Date"], inplace=True)
    df["SP500_Return"] = df["SP500_Close"].pct_change()
    df["Bull_Bear_Spread"] = (df["Bullish"] - df["Bearish"]) * 100  # convert to %
    return df.dropna()

df = load_data()

st.sidebar.header("Strategy Parameters")
bull_threshold = st.sidebar.slider("Bullish Threshold (+%)", 0, 50, 20)
bear_threshold = st.sidebar.slider("Bearish Threshold (-%)", -50, 0, -20)
initial_capital = st.sidebar.number_input("Initial Capital ($)", 1000, 1000000, 10000, step=1000)

# Signal logic
df["Signal"] = 0
df.loc[df["Bull_Bear_Spread"] > bull_threshold, "Signal"] = 1
df.loc[df["Bull_Bear_Spread"] < bear_threshold, "Signal"] = -1
df["Position"] = df["Signal"].replace(to_replace=0, method="ffill")

# Performance

df["Market_Return"] = df["SP500_Close"].pct_change()
df["Strategy_Return"] = df["Market_Return"] * df["Position"].shift(1)
df.dropna(inplace=True)
df["Market_Cum"] = initial_capital * (1 + df["Market_Return"]).cumprod()
df["Strategy_Cum"] = initial_capital * (1 + df["Strategy_Return"]).cumprod()

# Plot results
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(df["Date"], df["Market_Cum"], label="Buy & Hold", color="gray")
ax.plot(df["Date"], df["Strategy_Cum"], label="Sentiment Strategy", color="green")
ax.set_title("Cumulative Return")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig)

# Performance metrics
strategy_total_return = df["Strategy_Cum"].iloc[-1] / initial_capital - 1
market_total_return = df["Market_Cum"].iloc[-1] / initial_capital - 1

st.markdown("### Performance Summary")
st.write(f"**Strategy Return:** {strategy_total_return:.2%}")
st.write(f"**Buy & Hold Return:** {market_total_return:.2%}")
