# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AAII Sentiment Risk App", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel("sentiment_data.xlsx", skiprows=3)

    # Rename the first columns to match our needs
    df = df.rename(columns={
        df.columns[0]: "Date",
        df.columns[1]: "Bullish",
        df.columns[2]: "Neutral",
        df.columns[3]: "Bearish",
        df.columns[-1]: "SP500_Close"  # assumes last column is weekly close
    })

    # Drop rows where Date or key values are missing
    df = df.dropna(subset=["Date", "Bullish", "Neutral", "Bearish", "SP500_Close"])

    # Remove any non-date rows (e.g., "Avg", "STD", etc.)
    df = df[df["Date"].apply(lambda x: isinstance(x, pd.Timestamp))]

    # Sort and process
    df = df.sort_values("Date").reset_index(drop=True)
    df[["Bullish", "Neutral", "Bearish"]] = df[["Bullish", "Neutral", "Bearish"]] * 100
    df["SP500_Return"] = df["SP500_Close"].pct_change() * 100

    return df.dropna()


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
