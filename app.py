import streamlit as st
import pandas as pd
import altair as alt
import datetime
import numpy as np
from scipy.stats import zscore


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
    "🧠 Deep Q-Learning Strategy"
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
# 📈 Tab 7: Deep Q-Learning Strategy with Optimized Features and Faster Runtime
with tab7:
    st.subheader("🧠 Deep Q-Learning Strategy")
    st.markdown("""
    This strategy uses Deep Q-Learning to learn an optimal trading policy based on sentiment and price momentum.

    - **State**: z-scores of bullish sentiment, bearish sentiment, bull-bear spread, 4-week price return, volatility, moving average, and lagged return  
    - **Actions**: -1 (short), 0 (neutral), 1 (long)  
    - **Reward**: Next week's return * action (reward shaping applied)  
    - **Training Range**: Selectable  
    - **Testing Range**: Selectable
    """)

    st.subheader("📅 Training and Testing Period")
    train_start = st.slider("Training Start Year", min_value=1987, max_value=2025, value=2000)
    train_end = st.slider("Training End Year", min_value=1987, max_value=2025, value=2015)

    # Filter and prepare data
    df_ml = clean_df.copy()
    df_ml = df_ml[df_ml['Date'].dt.year >= 1987]

    df_ml['Bull_Bear'] = df_ml['Bullish'] - df_ml['Bearish']
    df_ml['Return_4w'] = df_ml['SP500_Return'].rolling(4).sum()
    df_ml['Volatility'] = df_ml['SP500_Return'].rolling(4).std()
    df_ml['MA_4w'] = df_ml['SP500_Return'].rolling(4).mean()
    df_ml['Lagged_Return'] = df_ml['SP500_Return'].shift(1)
    df_ml = df_ml.dropna()

    df_ml['Bullish_z'] = zscore(df_ml['Bullish'])
    df_ml['Bearish_z'] = zscore(df_ml['Bearish'])
    df_ml['Bull_Bear_z'] = zscore(df_ml['Bull_Bear'])
    df_ml['Return_4w_z'] = zscore(df_ml['Return_4w'])

    features = ['Bullish_z', 'Bearish_z', 'Bull_Bear_z', 'Return_4w_z', 'Volatility', 'MA_4w', 'Lagged_Return']
    df_ml['Reward'] = df_ml['SP500_Return'].shift(-1)

    df_ml = df_ml.dropna()

    X = df_ml[features].values
    y = df_ml['Reward'].values
    dates = df_ml['Date']

    train_mask = (dates.dt.year >= train_start) & (dates.dt.year <= train_end)
    test_mask = (dates.dt.year > train_end)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_dates = dates[test_mask]

    # Reward shaping
    y_train_shaped = y_train.copy()
    y_train_shaped[y_train > 0] *= 1.5
    y_train_shaped[y_train < 0] *= 2.0

    valid_train = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train_shaped)
    valid_test = np.isfinite(X_test).all(axis=1) & np.isfinite(y_test)

    X_train, y_train_shaped = X_train[valid_train], y_train_shaped[valid_train]
    X_test, y_test = X_test[valid_test], y_test[valid_test]
    test_dates = test_dates[valid_test]

    model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=200, random_state=42)
    model.fit(X_train, y_train_shaped)

    q_values = model.predict(X_test)
    actions = np.where(q_values > 0.001, 1, np.where(q_values < -0.001, -1, 0))

    action_counts_train = dict(zip(*np.unique(np.where(y_train_shaped > 0, 1, np.where(y_train_shaped < 0, -1, 0)), return_counts=True)))
    st.subheader("🧠 Training Action Distribution:")
    st.write(action_counts_train)

    portfolio = [10000]
    bh_portfolio = [10000]
    for i in range(len(y_test)):
        portfolio.append(portfolio[-1] * (1 + y_test[i] * actions[i]))
        bh_portfolio.append(bh_portfolio[-1] * (1 + y_test[i]))

    portfolio = portfolio[1:]
    bh_portfolio = bh_portfolio[1:]

    perf_df = pd.DataFrame({
        'Date': test_dates,
        'Q_Portfolio': portfolio,
        'BuyHold': bh_portfolio
    })

    chart = alt.Chart(perf_df).mark_line().encode(
        x='Date:T',
        y=alt.Y('Q_Portfolio:Q', title='Portfolio Value'),
        color=alt.value("#1f77b4")
    ).properties(title="Q-Learning vs Buy & Hold")

    bh_line = alt.Chart(perf_df).mark_line(strokeDash=[4, 4]).encode(
        x='Date:T',
        y='BuyHold:Q',
        color=alt.value("#aec7e8")
    )

    st.altair_chart(chart + bh_line, use_container_width=True)

    q_return = (portfolio[-1] / portfolio[0]) - 1 if portfolio[0] > 0 else 0
    bh_return = (bh_portfolio[-1] / bh_portfolio[0]) - 1 if bh_portfolio[0] > 0 else 0

    st.subheader("📊 Performance Summary")
    st.markdown(f"""
    - **Deep Q-Learning Strategy Return**: {q_return:.2%}  
    - **Buy & Hold Return**: {bh_return:.2%}
    """)

    test_action_counts = dict(zip(*np.unique(actions, return_counts=True)))
    st.markdown("**Action Distribution (Test Set):**")
    st.write({"Long": test_action_counts.get(1, 0), "Short": test_action_counts.get(-1, 0), "Neutral": test_action_counts.get(0, 0)})
