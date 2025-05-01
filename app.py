import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime

# === Load and clean data ===
@st.cache_data

def load_data():
    file_path = "sentiment_data.xlsx"
    df = pd.read_excel(file_path, skiprows=4)
    df = df.rename(columns={
        df.columns[0]: "Date",
        df.columns[1]: "Bullish",
        df.columns[2]: "Neutral",
        df.columns[3]: "Bearish",
        df.columns[12]: "SP500_Close"
    })
    df = df[pd.to_datetime(df['Date'], errors='coerce').notna()]
    df['Date'] = pd.to_datetime(df['Date'])
    df[['Bullish', 'Neutral', 'Bearish']] = df[['Bullish', 'Neutral', 'Bearish']].apply(pd.to_numeric, errors='coerce')
    df['SP500_Close'] = pd.to_numeric(df['SP500_Close'], errors='coerce')
    df = df.dropna(subset=['SP500_Close'])
    df = df.sort_values('Date')
    return df

df = load_data()

# === Compute weekly returns ===
df['Return'] = df['SP500_Close'].pct_change() * 100

# === Regression ===
df['Bullish_lag1'] = df['Bullish'].shift(1)
df['Bearish_lag1'] = df['Bearish'].shift(1)
reg_data = df.dropna(subset=['Return', 'Bullish_lag1', 'Bearish_lag1'])

X = reg_data[['Bullish_lag1', 'Bearish_lag1']]
X = sm.add_constant(X)
y = reg_data['Return']

model = sm.OLS(y, X).fit()
reg_data['Predicted_Return'] = model.predict(X)
reg_data['Residuals'] = reg_data['Return'] - reg_data['Predicted_Return']

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("📉 AAII Sentiment & S&P 500 Dashboard")

with st.expander("📊 Factor Model: Estimating Factor Loadings via Regression", expanded=True):
    st.markdown("""
    **Factor Model:** Estimating how much S&P 500 weekly returns are explained by investor sentiment.
    
    Model:  
    $R_t = \alpha + \beta_1 \cdot \text{Bullish}_{t-1} + \beta_2 \cdot \text{Bearish}_{t-1} + \varepsilon_t$
    
    In matrix form: $X = F \cdot B + E \Rightarrow \hat{B} = (F^\top F)^{-1}F^\top X$
    """)

    # Show regression summary
    st.text(model.summary())

    # Plot actual vs predicted returns
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(reg_data['Date'], reg_data['Return'], color='black', label='Actual', linewidth=0.5)
    ax1.plot(reg_data['Date'], reg_data['Predicted_Return'], color='orange', linestyle='--', label='Predicted', linewidth=1.2)
    ax1.set_title("Actual vs Predicted S&P 500 Weekly Returns")
    ax1.set_ylabel("Weekly Return (%)")
    ax1.legend()
    ax1.grid(True, linestyle=':', linewidth=0.5)
    st.pyplot(fig1)

    # Plot residual distribution
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.hist(reg_data['Residuals'], bins=50, color='gray', edgecolor='black')
    ax2.set_title("Distribution of Residuals")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, linestyle=':', linewidth=0.5)
    st.pyplot(fig2)

    # Show R^2
    r2 = model.rsquared
    st.markdown(f"""
    **$R^2$ (Explained Variance):** {r2:.3f}  
    The $R^2$ value indicates the portion of return variance explained by sentiment factors (systematic risk). 
    Remaining variation is attributed to residual (idiosyncratic) noise.
    """)
