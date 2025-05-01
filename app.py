import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

st.set_page_config(page_title="AAII Sentiment Risk App", layout="wide")

st.title("\U0001F4CA AAII Sentiment & S&P 500 Dashboard")

@st.cache_data
def load_raw_excel():
    return pd.read_excel("sentiment_data.xlsx", header=None)

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

raw_df = load_raw_excel()
clean_df = load_clean_data()

tab1, tab2, tab3 = st.tabs([
    "\U0001F5C2 Raw Excel Viewer",
    "\U0001F4C8 Interactive Dashboard",
    "\U0001F52C Factor Model"
])

with tab1:
    st.header("\U0001F5C2 Raw AAII Sentiment Excel File")
    st.dataframe(raw_df)

with tab2:
    st.header("\U0001F4C6 Select Time Range for Analysis")

    min_date = clean_df["Date"].min().date()
    max_date = clean_df["Date"].max().date()

    start_date, end_date = st.slider("Select a date range:", min_value=min_date, max_value=max_date, value=(min_date, max_date), format="YYYY-MM-DD")
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = clean_df[(clean_df["Date"] >= start_date) & (clean_df["Date"] <= end_date)]

    st.markdown("##### \U0001F4C9 S&P 500 Weekly Close (Log Scale)")
    fig1, ax1 = plt.subplots(figsize=(10, 2))
    ax1.plot(filtered_df["Date"], filtered_df["SP500_Close"], color="black", linewidth=0.8)
    ax1.set_yscale("log")
    ax1.set_ylabel("Price", fontsize=8)
    ax1.tick_params(axis='both', labelsize=7)
    ax1.grid(True, linestyle="--", linewidth=0.25, alpha=0.5)
    st.pyplot(fig1)

    st.markdown("##### \U0001F9E0 Investor Sentiment (Toggle Lines)")
    col1, col2, col3 = st.columns(3)
    show_bullish = col1.checkbox("\U0001F402 Bullish", value=True)
    show_neutral = col2.checkbox("☰ Neutral", value=True)
    show_bearish = col3.checkbox("\U0001F43B Bearish", value=True)

    fig2, ax2 = plt.subplots(figsize=(10, 2))
    if show_bullish:
        ax2.plot(filtered_df["Date"], filtered_df["Bullish"], label="Bullish", color="green", linewidth=0.8)
    if show_neutral:
        ax2.plot(filtered_df["Date"], filtered_df["Neutral"], label="Neutral", color="gray", linewidth=0.8)
    if show_bearish:
        ax2.plot(filtered_df["Date"], filtered_df["Bearish"], label="Bearish", color="red", linewidth=0.8)
    ax2.set_ylabel("Sentiment (%)", fontsize=8)
    ax2.tick_params(axis='both', labelsize=7)
    ax2.legend(fontsize=7, loc="upper left", frameon=False)
    ax2.grid(True, linestyle="--", linewidth=0.25, alpha=0.5)
    st.pyplot(fig2)

    st.markdown("##### \U0001F4C8 Bullish Sentiment Moving Average")
    ma_window = st.slider("Select MA Window (weeks):", 1, 52, 4, key="tab2_ma")

    df_ma = filtered_df.copy()
    df_ma["Bullish_MA"] = df_ma["Bullish"].rolling(window=ma_window, min_periods=1).mean()

    fig3, ax3 = plt.subplots(figsize=(10, 2))
    ax3.plot(df_ma["Date"], df_ma["SP500_Close"], color="black", label="S&P 500", linewidth=0.8)
    ax4 = ax3.twinx()
    ax4.plot(df_ma["Date"], df_ma["Bullish_MA"], color="green", label=f"Bullish ({ma_window}-W MA)", linewidth=0.8)
    ax3.set_ylabel("S&P 500 Price", fontsize=8, color="black")
    ax4.set_ylabel("Bullish Sentiment (%)", fontsize=8, color="green")
    ax3.tick_params(axis='both', labelsize=7, labelcolor="black")
    ax4.tick_params(axis='both', labelsize=7, labelcolor="green")
    ax3.grid(True, linestyle="--", linewidth=0.25, alpha=0.5)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7, frameon=False)
    st.pyplot(fig3)

    st.subheader("\U0001F4CB Filtered Data Table")
    st.dataframe(filtered_df, use_container_width=True, height=400)

with tab3:
    st.header("\U0001F52C Factor Model: Estimating Factor Loadings via Regression")

    st.markdown("""
    This model estimates how weekly S&P 500 returns respond to observable sentiment factors.
    We apply a classical regression-based calibration as described in the factor model literature.

    **Model:**
    \[ R_t = \alpha + \beta_1 \cdot \text{Bullish}_t + \beta_2 \cdot \text{Bearish}_t + \varepsilon_t \]

    In matrix form:
    \[ X = F B + E, \quad \hat{B} = (F^\top F)^{-1} F^\top X \]
    """)

    reg_data = filtered_df[["Bullish", "Bearish", "SP500_Return"]].dropna()
    X = reg_data[["Bullish", "Bearish"]]
    X = sm.add_constant(X)
    y = reg_data["SP500_Return"]

    model = sm.OLS(y, X).fit()

    st.markdown("### Regression Summary")
    st.text(model.summary())

    st.markdown("### Estimated Factor Loadings (\u03B2)")
    fig_coef, ax = plt.subplots(figsize=(6, 2))
    model.params[1:].plot(kind="bar", ax=ax, color=["green", "red"], width=0.6)
    ax.set_ylabel("Coefficient", fontsize=8)
    ax.set_ylim(-10, 10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.25, alpha=0.5)
    st.pyplot(fig_coef)

    st.markdown(f"**R² (Explained Variance)**: {model.rsquared:.3f}")
    st.markdown("The R² value indicates the portion of return variance explained by sentiment factors (systematic risk). Remaining variation is attributed to residual (idiosyncratic) noise.")
