import streamlit as st
import pandas as pd
import altair as alt
import datetime

st.set_page_config(page_title="AAII Sentiment Dashboard (Altair)", layout="wide")

st.title(":bar_chart: AAII Sentiment & S&P 500 Dashboard")

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

tab1, tab2, tab3 = st.tabs([":file_folder: Raw Excel Viewer", ":chart_with_upwards_trend: Interactive Dashboard", ":bar_chart: Sentiment Strategy"])

# ---------------------------- TAB 1 ----------------------------------
with tab1:
    st.header(":file_folder: Raw AAII Sentiment Excel File")
    st.dataframe(raw_df)

# ---------------------------- TAB 2 ----------------------------------
with tab2:
    st.header(":calendar: Time Range Selection")

    min_date = clean_df["Date"].min().date()
    max_date = clean_df["Date"].max().date()
    start_date, end_date = st.slider("Select a date range:", min_value=min_date, max_value=max_date, value=(min_date, max_date), format="YYYY-MM-DD")
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = clean_df[(clean_df["Date"] >= start_date) & (clean_df["Date"] <= end_date)]

    # Chart 1: S&P 500 log price (Altair)
    st.subheader(":label: S&P 500 Weekly Close (Log Scale)")
    log_chart = alt.Chart(filtered_df).mark_line(color="black").encode(
        x="Date:T",
        y=alt.Y("SP500_Close:Q", scale=alt.Scale(type="log"), title="Price"),
        tooltip=["Date:T", "SP500_Close"]
    ).properties(width=800, height=300).interactive()
    st.altair_chart(log_chart, use_container_width=True)

    # Chart 2: Sentiment lines (Altair)
    st.subheader(":brain: Investor Sentiment (Toggle Lines)")

    show_bullish = st.checkbox(":cow: Bullish", value=True)
    show_neutral = st.checkbox("\u2261 Neutral", value=True)
    show_bearish = st.checkbox(":bear: Bearish", value=True)

    sentiment_long = filtered_df.melt(id_vars=["Date"], value_vars=["Bullish", "Neutral", "Bearish"], var_name="Sentiment", value_name="Value")
    sentiment_long = sentiment_long[
        ((sentiment_long["Sentiment"] == "Bullish") & show_bullish) |
        ((sentiment_long["Sentiment"] == "Neutral") & show_neutral) |
        ((sentiment_long["Sentiment"] == "Bearish") & show_bearish)
    ]

    color_map = {"Bullish": "green", "Neutral": "gray", "Bearish": "red"}
    sentiment_chart = alt.Chart(sentiment_long).mark_line().encode(
        x="Date:T",
        y=alt.Y("Value:Q", title="Sentiment (%)"),
        color=alt.Color("Sentiment:N", scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values()))),
        tooltip=["Date:T", "Sentiment:N", "Value:Q"]
    ).properties(width=800, height=300).interactive()

    st.altair_chart(sentiment_chart, use_container_width=True)

# ---------------------------- TAB 3 ----------------------------------
with tab3:
    st.header(":test_tube: Z-Score Strategy Backtest")

    window = st.slider("Rolling Window (days)", min_value=5, max_value=60, value=15)
    capital = st.number_input("Initial Capital ($)", value=10000)

    df = clean_df.copy().set_index("Date")
    df["Z_bullish"] = (df["Bullish"] - df["Bullish"].rolling(window).mean()) / df["Bullish"].rolling(window).std()
    df["Z_price"] = (df["SP500_Close"] - df["SP500_Close"].rolling(window).mean()) / df["SP500_Close"].rolling(window).std()
    df.dropna(inplace=True)

    df["Position"] = (df["Z_bullish"] > df["Z_price"]).astype(int) * 2 - 1
    df["Strategy_Return"] = df["SP500_Return"] * df["Position"] / 100

    df["BuyHold"] = (1 + df["SP500_Return"] / 100).cumprod() * capital
    df["Strategy"] = (1 + df["Strategy_Return"]).cumprod() * capital

    performance_chart = alt.Chart(df.reset_index()).transform_fold([
        "BuyHold", "Strategy"]
    ).mark_line().encode(
        x="Date:T",
        y=alt.Y("value:Q", title="Portfolio Value ($)"),
        color=alt.Color("key:N", scale=alt.Scale(domain=["BuyHold", "Strategy"], range=["gray", "blue"])),
        tooltip=["Date:T", "key:N", "value:Q"]
    ).properties(title="Cumulative Return", width=800, height=300).interactive()

    st.altair_chart(performance_chart, use_container_width=True)

    st.markdown("**Performance Summary**")
    strat_return = df["Strategy"].iloc[-1] / capital - 1
    bh_return = df["BuyHold"].iloc[-1] / capital - 1
    st.write(f"Strategy Return: {strat_return:.2%}")
    st.write(f"Buy & Hold Return: {bh_return:.2%}")
