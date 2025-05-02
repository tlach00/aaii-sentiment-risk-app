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
    st.markdown("## :chart_with_upwards_trend: Interactive Dashboard")

    min_date = clean_df["Date"].min().date()
    max_date = clean_df["Date"].max().date()

    start_date, end_date = st.slider(
        "Select a date range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = clean_df[(clean_df["Date"] >= start_date) & (clean_df["Date"] <= end_date)]

    # S&P 500 Chart
    st.markdown("### :newspaper: S&P 500 Weekly Close (Log Scale)")
    chart1 = alt.Chart(filtered_df).mark_line(color='black').encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('SP500_Close:Q', scale=alt.Scale(type='log'), title='Price')
    ).properties(
        height=300
    )
    st.altair_chart(chart1, use_container_width=True)

    # Sentiment Toggle
    st.markdown("### 🧠 Investor Sentiment (Toggle Lines)")
    col1, col2, col3 = st.columns(3)
    show_bullish = col1.checkbox("🐂 Bullish", value=True)
    show_neutral = col2.checkbox("≡ Neutral", value=True)
    show_bearish = col3.checkbox("🐻 Bearish", value=True)

    chart2 = alt.Chart(filtered_df).transform_fold(
        ["Bullish", "Neutral", "Bearish"],
        as_=["Sentiment", "Value"]
    )
    chart2 = chart2.mark_line().encode(
        x='Date:T',
        y=alt.Y('Value:Q', title='Sentiment (%)'),
        color=alt.Color('Sentiment:N', scale=alt.Scale(domain=["Bullish", "Neutral", "Bearish"],
                                                       range=["green", "gray", "red"]))
    )

    filters = []
    if show_bullish: filters.append("Bullish")
    if show_neutral: filters.append("Neutral")
    if show_bearish: filters.append("Bearish")

    chart2 = chart2.transform_filter(alt.FieldOneOfPredicate(field='Sentiment', oneOf=filters))

    st.altair_chart(chart2, use_container_width=True)

    # Superposed Bullish MA vs SP500
    st.markdown("### :chart_with_upwards_trend: Bullish Sentiment Moving Average")
    ma_window = st.slider("Select MA Window (weeks):", 1, 52, 52, key="tab2_ma")

    df_ma = filtered_df.copy()
    df_ma["Bullish_MA"] = df_ma["Bullish"].rolling(window=ma_window, min_periods=1).mean()

    base = alt.Chart(df_ma).encode(x='Date:T')

    chart3 = alt.layer(
        base.mark_line(color='black').encode(y=alt.Y('SP500_Close:Q', title='S&P 500 Price')),
        base.mark_line(color='green').encode(y=alt.Y('Bullish_MA:Q', title='Bullish Sentiment MA'))
    ).resolve_scale(y='independent').properties(height=300)

    st.altair_chart(chart3, use_container_width=True)

    # Table
    st.markdown("### :clipboard: Filtered Data Table")
    st.dataframe(filtered_df, use_container_width=True, height=400)

# ---------------------------- TAB 3 ----------------------------------
with tab3:
    st.markdown("""
        ## :bar_chart: Z-Score Strategy Backtest
    """)

    st.markdown("""Customize the strategy inputs:
    """)

    # --- Sidebar Parameters ---
    col1, col2 = st.columns(2)

    with col1:
        z_window = st.slider("Rolling Window (days)", min_value=5, max_value=60, value=52)

    with col2:
        initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=10000, step=100)

    # --- Prepare Data for Backtest ---
    df_z = clean_df.copy()
    df_z = df_z.set_index("Date")
    df_z = df_z["1988":]  # S&P recovery post crash

    # --- Compute Z-scores ---
    df_z["Bullish_Z"] = (df_z["Bullish"] - df_z["Bullish"].rolling(z_window).mean()) / df_z["Bullish"].rolling(z_window).std()
    df_z["Price_Z"] = (df_z["SP500_Close"] - df_z["SP500_Close"].rolling(z_window).mean()) / df_z["SP500_Close"].rolling(z_window).std()

    df_z.dropna(inplace=True)

    # --- Strategy Logic ---
    df_z["Position"] = 0
    df_z.loc[df_z["Bullish_Z"] > df_z["Price_Z"], "Position"] = 1
    df_z.loc[df_z["Bullish_Z"] < df_z["Price_Z"], "Position"] = -1

    df_z["Return"] = df_z["SP500_Return"] / 100
    df_z["Strategy_Return"] = df_z["Return"] * df_z["Position"]

    # --- Cumulative Portfolio Value ---
    df_z["BuyHold_Portfolio"] = (1 + df_z["Return"]).cumprod() * initial_capital
    df_z["Strategy_Portfolio"] = (1 + df_z["Strategy_Return"]).cumprod() * initial_capital

    # --- Plot ---
    st.markdown("### Cumulative Return")
    import altair as alt

    df_plot = df_z.reset_index()[["Date", "BuyHold_Portfolio", "Strategy_Portfolio"]].melt("Date", var_name="Strategy", value_name="Value")
    chart = alt.Chart(df_plot).mark_line().encode(
        x="Date:T",
        y=alt.Y("Value:Q", title="Portfolio Value ($)"),
        color=alt.Color("Strategy", scale=alt.Scale(domain=["BuyHold_Portfolio", "Strategy_Portfolio"],
                                                     range=["#6c6c6c", "#0070f3"]),
                      legend=alt.Legend(title=None, labelExpr="replace(datum.label, '_Portfolio', '')"))
    ).properties(height=350)
    st.altair_chart(chart, use_container_width=True)

    # --- Performance Summary ---
    strat_ret = (df_z["Strategy_Portfolio"].iloc[-1] / initial_capital - 1) * 100
    bh_ret = (df_z["BuyHold_Portfolio"].iloc[-1] / initial_capital - 1) * 100

    st.markdown("""
        ### Performance Summary
    """)
    st.write(f"**Strategy Return:** {strat_ret:.2f}%")
    st.write(f"**Buy & Hold Return:** {bh_ret:.2f}%")

