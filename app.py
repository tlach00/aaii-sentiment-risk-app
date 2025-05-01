# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AAII Sentiment Risk App", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel("sentiment_data.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Date", "Bullish", "Neutral", "Bearish", "SP500_Close"])
    df = df.sort_values("Date").reset_index(drop=True)
    df[["Bullish", "Neutral", "Bearish"]] *= 100
    df["SP500_Return"] = df["SP500_Close"].pct_change() * 100
    return df.dropna()

df = load_data()

# --- Sidebar ---
st.sidebar.header("AAII Sentiment Dashboard")
st.sidebar.markdown("Data source: [AAII](https://www.aaii.com/sentimentsurvey)")

# --- Main View ---
st.title("📊 Market Sentiment & S&P 500 Risk Model")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Weekly AAII Sentiment")
    st.line_chart(df.set_index("Date")[["Bullish", "Neutral", "Bearish"]])

with col2:
    st.subheader("S&P 500 Weekly Returns (%)")
    st.line_chart(df.set_index("Date")["SP500_Return"])

# --- Regression Model ---
st.subheader("Factor Model: Explaining S&P 500 Returns with Sentiment")

import statsmodels.api as sm

X = df[["Bullish", "Neutral", "Bearish"]]
X = sm.add_constant(X)  # adds intercept
y = df["SP500_Return"]
model = sm.OLS(y, X).fit()

st.markdown("**Regression Summary:**")
st.text(model.summary())

# --- Coefficient Plot ---
st.subheader("Sentiment Factor Coefficients")

fig, ax = plt.subplots()
coef = model.params[1:]  # exclude intercept
coef.plot(kind="bar", ax=ax)
ax.set_ylabel("Coefficient")
st.pyplot(fig)
