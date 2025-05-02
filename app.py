import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler

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
    "\U0001F4C9 Sentiment-Based Backtest Strategy"
])

# TAB 1 - Raw Excel Viewer
with tab1:
    st.header("\U0001F5C2 Raw AAII Sentiment Excel File")
    st.dataframe(raw_df)

# TAB 2 - Interactive Dashboard
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
    show_neutral = col2.checkbox("\u2630 Neutral", value=True)
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

# TAB 3 - Placeholder for future strategy implementation
with tab3:
    st.header("📈 Sentiment-Based Backtest Strategy")

    st.sidebar.markdown("### Strategy Parameters")
    bullish_threshold = st.sidebar.slider("Bullish Threshold (%)", 0.0, 100.0, 50.0, step=1.0)
    bearish_threshold = st.sidebar.slider("Bearish Threshold (%)", 0.0, 100.0, 50.0, step=1.0)
    initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=10000, step=1000)

    df = clean_df.copy()
    df = df.set_index("Date")

    # Create signal: 1 = invest, 0 = cash
    df["Signal"] = 0
    df.loc[df["Bullish"] >= bullish_threshold / 100, "Signal"] = 1
    df.loc[df["Bearish"] >= bearish_threshold / 100, "Signal"] = 0

    # Shift signals to simulate decision at prior week
    df["Signal"] = df["Signal"].shift(1).fillna(0)

    # Calculate strategy returns
    df["Strategy_Return"] = df["Signal"] * df["SP500_Return"]
    df["BuyHold_Return"] = df["SP500_Return"]

    # Calculate cumulative capital over time
    df["Strat_Value"] = initial_capital * (1 + df["Strategy_Return"] / 100).cumprod()
    df["BH_Value"] = initial_capital * (1 + df["BuyHold_Return"] / 100).cumprod()

    # Plot cumulative returns
    st.markdown("### Cumulative Return")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["BH_Value"], label="Buy & Hold", color="gray", linewidth=1.2)
    ax.plot(df.index, df["Strat_Value"], label="Sentiment Strategy", color="green", linewidth=1.2)
    ax.set_title("Cumulative Return", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontsize=9)
    ax.legend(fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.25, alpha=0.6)
    st.pyplot(fig)

    # Display performance metrics
    total_return_strategy = df["Strat_Value"].iloc[-1] / initial_capital - 1
    total_return_bh = df["BH_Value"].iloc[-1] / initial_capital - 1

    st.markdown("### Performance Summary")
    st.write(f"**Strategy Return**: {total_return_strategy:.2%}")
    st.write(f"**Buy & Hold Return**: {total_return_bh:.2%}")
