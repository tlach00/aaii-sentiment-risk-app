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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    ":file_folder: Raw Excel Viewer",
    ":chart_with_upwards_trend: Interactive Dashboard",
    "🧪 Z-Score Strategy Backtest",
    "🟥 Z-Score Spread Strategy",
    "📊 Weighted Allocation Strategy",
    "🧬 Multi-Factor Strategy",
    "🤖 Q-Learning Strategy"
])

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

    st.markdown("### :newspaper: S&P 500 Weekly Close (Log Scale)")
    chart1 = alt.Chart(filtered_df).mark_line(color='black').encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('SP500_Close:Q', scale=alt.Scale(type='log'), title='Price')
    ).properties(height=300)
    st.altair_chart(chart1, use_container_width=True)

    st.markdown("### 🧠 Investor Sentiment (Toggle Lines)")
    col1, col2, col3 = st.columns(3)
    show_bullish = col1.checkbox("🐂 Bullish", value=True)
    show_neutral = col2.checkbox("≡ Neutral", value=True)
    show_bearish = col3.checkbox("🐻 Bearish", value=True)

    chart2 = alt.Chart(filtered_df).transform_fold(
        ["Bullish", "Neutral", "Bearish"],
        as_=["Sentiment", "Value"]
    ).mark_line().encode(
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

    st.markdown("### :clipboard: Filtered Data Table")
    st.dataframe(filtered_df, use_container_width=True, height=400)

# ---------------------------- TAB 3 ----------------------------------
with tab3:
    st.header("🟥 Z-Score Spread Strategy vs. Buy & Hold")
    st.markdown("This strategy compares z-scores of bullish sentiment and price. When sentiment is stronger (Z_Bullish > Z_Price), we go long. Otherwise, we short.")

    col1, col2 = st.columns(2)
    with col1:
        spread_window = st.slider("Z-Score Rolling Window (weeks)", 1, 5, 2)
    with col2:
        base_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)

    spread_df = clean_df.copy().set_index("Date").dropna()

    spread_df["Z_Bullish"] = (spread_df["Bullish"] - spread_df["Bullish"].rolling(spread_window).mean()) / spread_df["Bullish"].rolling(spread_window).std()
    spread_df["Z_Price"] = (spread_df["SP500_Close"] - spread_df["SP500_Close"].rolling(spread_window).mean()) / spread_df["SP500_Close"].rolling(spread_window).std()
    spread_df = spread_df.dropna()

    spread_df["Signal"] = (spread_df["Z_Bullish"] > spread_df["Z_Price"]).astype(int) * 2 - 1
    spread_df["Position"] = spread_df["Signal"].shift(1).fillna(0)

    spread_df["SP500_Ret"] = spread_df["SP500_Return"] / 100
    spread_df["Strategy_Ret"] = spread_df["Position"] * spread_df["SP500_Ret"]

    spread_df["BuyHold"] = (1 + spread_df["SP500_Ret"]).cumprod() * base_capital
    spread_df["ZSpread"] = (1 + spread_df["Strategy_Ret"]).cumprod() * base_capital

    chart = alt.Chart(spread_df.reset_index()).transform_fold(
        ["BuyHold", "ZSpread"]
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
# ---------------------------- TAB 4 ----------------------------------

with tab4:
    st.markdown("## :arrow_up_down: Sentiment Momentum Strategy")
    st.markdown("This strategy compares short-term vs long-term moving averages of bullish sentiment. If short-term > long-term → long, otherwise → short.")

    col1, col2 = st.columns(2)
    with col1:
        short_window = st.slider("Short-Term MA (weeks)", min_value=1, max_value=10, value=2)
    with col2:
        long_window = st.slider("Long-Term MA (weeks)", min_value=5, max_value=52, value=15)

    df_mom = clean_df.copy().set_index("Date")
    df_mom = df_mom[['Bullish', 'SP500_Close', 'SP500_Return']].dropna()

    df_mom['MA_short'] = df_mom['Bullish'].rolling(window=short_window).mean()
    df_mom['MA_long'] = df_mom['Bullish'].rolling(window=long_window).mean()

    df_mom = df_mom.dropna()

    df_mom['Signal'] = (df_mom['MA_short'] > df_mom['MA_long']).astype(int) * 2 - 1
    df_mom['Position'] = df_mom['Signal'].shift(1).fillna(0)

    df_mom['SP500_Ret'] = df_mom['SP500_Return'] / 100
    df_mom['Strategy_Ret'] = df_mom['Position'] * df_mom['SP500_Ret']

    initial_mom_capital = 10000
    df_mom['Momentum'] = (1 + df_mom['Strategy_Ret']).cumprod() * initial_mom_capital
    df_mom['BuyHold'] = (1 + df_mom['SP500_Ret']).cumprod() * initial_mom_capital

    mom_chart = alt.Chart(df_mom.reset_index()).transform_fold([
        "BuyHold", "Momentum"]
    ).mark_line().encode(
        x="Date:T",
        y=alt.Y("value:Q", title="Portfolio Value ($)"),
        color=alt.Color("key:N", title="Strategy")
    ).properties(height=350)

    st.altair_chart(mom_chart, use_container_width=True)

    strat_ret = df_mom["Momentum"].iloc[-1] / initial_mom_capital - 1
    bh_ret = df_mom["BuyHold"].iloc[-1] / initial_mom_capital - 1

    st.markdown(f"""
    ### 📊 Performance Summary
    - **Momentum Strategy Return:** {strat_ret:.2%}  
    - **Buy & Hold Return:** {bh_ret:.2%}
    """)
# ---------------------------- TAB 5 ----------------------------------
with tab5:
    st.markdown("### 📊 Weighted Allocation Strategy (No Smoothing)")

    st.markdown("""
    This strategy allocates capital based on the weekly difference between bullish and bearish sentiment.  
    \n**Position = Bullish% - Bearish%**
    
    - If bullish sentiment exceeds bearish → positive position (long exposure)  
    - If bearish exceeds bullish → negative position (short exposure)  
    - No smoothing is applied — sentiment is used directly week-to-week.
    """)

    capital_w = st.number_input("Initial Capital ($)", value=10000, step=1000, key="tab5_nosmooth")

    df_weight_raw = clean_df.copy().set_index("Date")
    df_weight_raw["Position"] = df_weight_raw["Bullish"] - df_weight_raw["Bearish"]
    df_weight_raw["SP500_Ret"] = df_weight_raw["SP500_Return"] / 100

    df_weight_raw["Strategy_Ret"] = df_weight_raw["Position"].shift(1) * df_weight_raw["SP500_Ret"]

    df_weight_raw["BuyHold"] = (1 + df_weight_raw["SP500_Ret"]).cumprod() * capital_w
    df_weight_raw["Weighted"] = (1 + df_weight_raw["Strategy_Ret"]).cumprod() * capital_w

    chart_ws = alt.Chart(df_weight_raw.reset_index()).transform_fold([
        "BuyHold", "Weighted"]
    ).mark_line().encode(
        x="Date:T",
        y=alt.Y("value:Q", title="Portfolio Value ($)"),
        color=alt.Color("key:N", title="Strategy")
    ).properties(height=350)

    st.altair_chart(chart_ws, use_container_width=True)

    strat_ret = df_weight_raw["Weighted"].iloc[-1] / capital_w - 1
    bh_ret = df_weight_raw["BuyHold"].iloc[-1] / capital_w - 1

    st.markdown(f"""
    #### 📈 Performance Summary
    - **Weighted Strategy Return (No Smoothing):** {strat_ret:.2%}  
    - **Buy & Hold Return:** {bh_ret:.2%}
    """)


# ---------------------------- TAB 6 ----------------------------------
with tab6:
    st.markdown("### 🧬 Multi-Factor Strategy (Z-Score + Spread + Momentum)")

    st.markdown("""
    This strategy combines:
    - Z-score of Bullish Sentiment  
    - Bull-Bear Sentiment Spread  
    - S&P 500 Momentum (4-week return)  
    \nEach component is standardized and combined into a single signal.
    If the signal > 0 → go long; else → go short.
    """)

    capital_mf = st.number_input("Initial Capital ($)", value=10000, step=1000, key="tab6_mf")

    df_mf = clean_df.copy().set_index("Date")
    df_mf = df_mf.dropna()
    
    # Momentum: 4-week return of S&P 500
    df_mf["Momentum"] = df_mf["SP500_Close"].pct_change(4)

    # Sentiment spread
    df_mf["Spread"] = df_mf["Bullish"] - df_mf["Bearish"]

    # Z-score of Bullish
    z_window = 15
    df_mf["Z_Bullish"] = (df_mf["Bullish"] - df_mf["Bullish"].rolling(z_window).mean()) / df_mf["Bullish"].rolling(z_window).std()

    # Normalize each component to z-score
    for col in ["Spread", "Momentum"]:
        df_mf[f"Z_{col}"] = (df_mf[col] - df_mf[col].rolling(z_window).mean()) / df_mf[col].rolling(z_window).std()

    df_mf = df_mf.dropna()

    # Signal = linear combo of 3 z-scores
    df_mf["Signal"] = df_mf["Z_Bullish"] + df_mf["Z_Spread"] + df_mf["Z_Momentum"]

    # Directional position: +1 (long), -1 (short)
    df_mf["Position"] = df_mf["Signal"].apply(lambda x: 1 if x > 0 else -1)

    df_mf["SP500_Ret"] = df_mf["SP500_Return"] / 100
    df_mf["Strat_Ret"] = df_mf["Position"].shift(1) * df_mf["SP500_Ret"]

    df_mf["BuyHold"] = (1 + df_mf["SP500_Ret"]).cumprod() * capital_mf
    df_mf["MultiFactor"] = (1 + df_mf["Strat_Ret"]).cumprod() * capital_mf

    chart_mf = alt.Chart(df_mf.reset_index()).transform_fold([
        "BuyHold", "MultiFactor"]
    ).mark_line().encode(
        x="Date:T",
        y=alt.Y("value:Q", title="Portfolio Value ($)"),
        color=alt.Color("key:N", title="Strategy")
    ).properties(height=350)

    st.altair_chart(chart_mf, use_container_width=True)

    mf_ret = df_mf["MultiFactor"].iloc[-1] / capital_mf - 1
    bh_ret = df_mf["BuyHold"].iloc[-1] / capital_mf - 1

    st.markdown(f"""
    #### 📈 Performance Summary
    - **Multi-Factor Strategy Return:** {mf_ret:.2%}  
    - **Buy & Hold Return:** {bh_ret:.2%}
    """)

# ---------------------------- TAB 7 ----------------------------------

with tab7:
    st.header("🤖 Sentiment Q-Learning Strategy")
    st.markdown("""
    This strategy uses Q-learning to learn an optimal trading policy based on sentiment and price momentum features.
    
    **States**: Discretized z-score of bullish/bearish sentiment, spread, and price momentum  
    **Actions**: -1 (short), 0 (neutral), 1 (long)  
    **Reward**: Next week's return * action
    
    **Training**: 1987 to 2014  
    **Testing**: 2015 to 2025
    """)

    # Parameters
    bins = 5
    gamma = 0.95
    alpha = 0.1
    episodes = 50
    capital = 10000

    # Data Prep
    df_rl = clean_df.copy().set_index("Date")
    df_rl = df_rl[['Bullish', 'Bearish', 'SP500_Close', 'SP500_Return']].dropna()

    df_rl['Spread'] = df_rl['Bullish'] - df_rl['Bearish']
    df_rl['Momentum'] = df_rl['SP500_Close'].pct_change(4)

    df_rl['Z_Bullish'] = (df_rl['Bullish'] - df_rl['Bullish'].rolling(15).mean()) / df_rl['Bullish'].rolling(15).std()
    df_rl['Z_Bearish'] = (df_rl['Bearish'] - df_rl['Bearish'].rolling(15).mean()) / df_rl['Bearish'].rolling(15).std()
    df_rl['Z_Spread'] = (df_rl['Spread'] - df_rl['Spread'].rolling(15).mean()) / df_rl['Spread'].rolling(15).std()
    df_rl['Z_Momentum'] = (df_rl['Momentum'] - df_rl['Momentum'].rolling(15).mean()) / df_rl['Momentum'].rolling(15).std()

    df_rl = df_rl.dropna()

    def discretize(series, bins):
        return pd.qcut(series, bins, labels=False, duplicates='drop')

    df_rl['s1'] = discretize(df_rl['Z_Bullish'], bins)
    df_rl['s2'] = discretize(df_rl['Z_Bearish'], bins)
    df_rl['s3'] = discretize(df_rl['Z_Spread'], bins)
    df_rl['s4'] = discretize(df_rl['Z_Momentum'], bins)

    df_rl['state'] = df_rl[['s1', 's2', 's3', 's4']].astype(str).agg('-'.join, axis=1)

    # Train/test split
    train_df = df_rl[df_rl.index < '2015-01-01']
    test_df = df_rl[df_rl.index >= '2015-01-01']

    q_table = {}
    actions = [-1, 0, 1]

    for _ in range(episodes):
        for i in range(len(train_df)-1):
            s = train_df['state'].iloc[i]
            a = np.random.choice(actions)
            r = train_df['SP500_Return'].iloc[i+1]/100 * a
            s_next = train_df['state'].iloc[i+1]

            if s not in q_table:
                q_table[s] = {a_: 0 for a_ in actions}
            if s_next not in q_table:
                q_table[s_next] = {a_: 0 for a_ in actions}

            best_next = max(q_table[s_next].values())
            q_table[s][a] += alpha * (r + gamma * best_next - q_table[s][a])

    test_df['Q_Action'] = test_df['state'].map(lambda s: max(q_table.get(s, {0: 0}), key=q_table.get(s, {0: 0}).get))
    test_df['BuyHold_Ret'] = test_df['SP500_Return'] / 100
    test_df['Q_Ret'] = test_df['BuyHold_Ret'] * test_df['Q_Action'].shift(1).fillna(0)

    test_df['BuyHold_Portfolio'] = (1 + test_df['BuyHold_Ret']).cumprod() * capital
    test_df['Q_Portfolio'] = (1 + test_df['Q_Ret']).cumprod() * capital

    line = alt.Chart(test_df.reset_index()).transform_fold([
        "BuyHold_Portfolio", "Q_Portfolio"]
    ).mark_line().encode(
        x="Date:T",
        y=alt.Y("value:Q", title="Portfolio Value ($)"),
        color=alt.Color("key:N", title="Strategy")
    ).properties(height=350)

    st.altair_chart(line, use_container_width=True)

    q_ret = test_df['Q_Portfolio'].iloc[-1] / capital - 1
    bh_ret = test_df['BuyHold_Portfolio'].iloc[-1] / capital - 1

    st.markdown(f"""
    ### 🧾 Performance Summary (2015–2025)
    - **Q-Learning Strategy Return**: {q_ret:.2%}  
    - **Buy & Hold Return**: {bh_ret:.2%}
    """)
