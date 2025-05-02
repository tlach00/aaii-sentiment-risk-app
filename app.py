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
    st.header("🧪 Z-Score Strategy Backtest")
    st.markdown("This strategy compares z-scores of bullish sentiment and S&P 500 price. If sentiment exceeds price → long, otherwise → short.")

    # Controls
    col1, col2 = st.columns([3, 2])
    with col1:
        window = st.slider("Rolling Window (days)", min_value=5, max_value=60, value=15)
    with col2:
        initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=10000, step=1000)

    df_z = clean_df.copy().set_index("Date")
    df_z = df_z[['Bullish', 'SP500_Close', 'SP500_Return']].dropna()

    # Calculate rolling z-scores
    df_z['Z_Bullish'] = (df_z['Bullish'] - df_z['Bullish'].rolling(window).mean()) / df_z['Bullish'].rolling(window).std()
    df_z['Z_Price'] = (df_z['SP500_Close'] - df_z['SP500_Close'].rolling(window).mean()) / df_z['SP500_Close'].rolling(window).std()

    # Only trade when both z-scores are available (avoids NaNs)
    df_z = df_z.dropna()

    # Trading logic: +1 for long, -1 for short
    df_z['Position'] = (df_z['Z_Bullish'] > df_z['Z_Price']).astype(int) * 2 - 1  # 1 or -1

    # Strategy returns
    df_z['Strategy_Return'] = df_z['Position'].shift(1) * df_z['SP500_Return'] / 100
    df_z['BuyHold_Return'] = df_z['SP500_Return'] / 100

    # Cumulative returns
    df_z['Portfolio'] = (1 + df_z['Strategy_Return']).cumprod() * initial_capital
    df_z['BuyHold'] = (1 + df_z['BuyHold_Return']).cumprod() * initial_capital

    st.subheader("📈 Cumulative Return")
    st.line_chart(df_z[['Portfolio', 'BuyHold']], height=360)

    st.subheader("🧮 Performance Summary")
    strategy_return = (df_z['Portfolio'].iloc[-1] / initial_capital - 1) * 100
    buyhold_return = (df_z['BuyHold'].iloc[-1] / initial_capital - 1) * 100

    st.markdown(f"""
    - **Strategy Return**: {strategy_return:.2f}%  
    - **Buy & Hold Return**: {buyhold_return:.2f}%
    """)
# ---------------------------- TAB 4 ----------------------------------

with tab4:
    st.markdown("""
    ### 🟥 Z-Score Spread Strategy vs. Buy & Hold
    This strategy compares z-scores of bullish sentiment and price.
    When sentiment is stronger (Z_Bullish > Z_Price), we go long.
    Otherwise, we short. Compare with a passive Buy & Hold strategy.
    """)

    col1, col2 = st.columns(2)
    with col1:
        spread_window = st.slider("Z-Score Rolling Window (weeks)", 1, 5, 2)
    with col2:
        base_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)

    spread_df = clean_df.copy().set_index("Date")
    spread_df = spread_df.dropna()

    # Compute z-scores
    spread_df["Z_Bullish"] = (spread_df["Bullish"] - spread_df["Bullish"].rolling(spread_window).mean()) / spread_df["Bullish"].rolling(spread_window).std()
    spread_df["Z_Price"] = (spread_df["SP500_Close"] - spread_df["SP500_Close"].rolling(spread_window).mean()) / spread_df["SP500_Close"].rolling(spread_window).std()

    spread_df = spread_df.dropna()

    # Strategy logic
    spread_df["Signal"] = (spread_df["Z_Bullish"] > spread_df["Z_Price"]).astype(int) * 2 - 1
    spread_df["Position"] = spread_df["Signal"].shift(1).fillna(0)

    spread_df["SP500_Ret"] = spread_df["SP500_Return"] / 100
    spread_df["Strategy_Ret"] = spread_df["Position"] * spread_df["SP500_Ret"]

    spread_df["BuyHold"] = (1 + spread_df["SP500_Ret"]).cumprod() * base_capital
    spread_df["ZSpread"] = (1 + spread_df["Strategy_Ret"]).cumprod() * base_capital

    chart = alt.Chart(spread_df.reset_index()).transform_fold([
        "BuyHold", "ZSpread"]
    ).mark_line().encode(
        x="Date:T",
        y=alt.Y("value:Q", title="Portfolio Value ($)"),
        color=alt.Color("key:N", title="Strategy")
    ).properties(height=350)

    st.altair_chart(chart, use_container_width=True)

    strat_ret = spread_df["ZSpread"].iloc[-1] / base_capital - 1
    bh_ret = spread_df["BuyHold"].iloc[-1] / base_capital - 1

    st.markdown(f"""
    #### 📈 Performance Summary
    - **Z-Score Strategy Return:** {strat_ret:.2%}  
    - **Buy & Hold Return:** {bh_ret:.2%}
    """)
