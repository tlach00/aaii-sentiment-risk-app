import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

st.set_page_config(page_title="AAII Sentiment Risk App", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel("sentiment_data.xlsx", skiprows=3)

    # Rename based on observed layout
    df = df.rename(columns={
        df.columns[0]: "Date",
        df.columns[1]: "Bullish",
        df.columns[2]: "Neutral",
        df.columns[3]: "Bearish",
        df.columns[-1]: "SP500_Close"  # Last column is weekly close
    })

    # Drop non-data rows
    df = df.dropna(subset=["Date", "Bullish", "Neutral", "Bearish", "SP500_Close"])
    df = df[df["Date"].apply(lambda x: isinstance(x, pd.Timestamp))]

    # Clean values
    df = df.sort_values("Date").reset_index(drop=True)
    df[["Bullish", "Neutral", "Bearish"]] = df[["Bullish", "Neutral", "Bearish"]] * 100
    df["SP500_Return"] = df["SP500_Close"].pct_change() * 100
    return df.dropna()

# Load data
df = load_data()

# --- Sidebar ---
st.sidebar.header("AAII Sentiment Dashboard")
st.sidebar.markdown("Data source: [AAII](https://www.aaii.com/sentimentsurvey)")

# --- Main title ---
st.title("📊 Market Sentiment & S&P 500 Risk Model")

# --- Line charts ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Weekly AAII Sentiment (%)")
    st.line_chart(df.set_index("Date")[["Bullish", "Neutral", "Bearish"]])

with col2:
    st.subheader("S&P 500 Weekly Returns (%)")
    st.line_chart(df.set_index("Date")["SP500_Return"])

# --- Factor model ---
st.subheader("Factor Model: Explaining S&P 500 Returns Using Sentiment")

X = df[["Bullish", "Neutral", "Bearish"]]
X = sm.add_constant(X)
y = df["SP500_Return"]
model = sm.OLS(y, X).fit()

st.markdown("### Regression Summary")
st.text(model.summary())

# --- Bar plot of coefficients ---
st.subheader("Sentiment Factor Coefficients")

fig, ax = plt.subplots()
model.params[1:].plot(kind="bar", ax=ax)  # Exclude intercept
ax.set_ylabel("Coefficient")
ax.set_title("Impact of Sentiment on S&P 500 Returns")
st.pyplot(fig)
