import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")
st.set_page_config(page_title="AAII Sentiment & S&P 500 Dashboard", layout="wide")
# Sklearn: Preprocessing, Model, Metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📁 Raw Excel Viewer",
    "📈 Interactive Dashboard",
    "🧪 Z-Score Strategy Backtest",
    "🟥 Z-Score Spread Strategy",
    "📊 Weighted Allocation Strategy",
    "🧬 Multi-Factor Strategy",
    "🧠 Deep Q-Learning Strategy",
    "📉 Fear & Greed Index",
    "😱 CNN F&G replication"
])

# ---------------------------- TAB 1 ----------------------------------
with tab1:
    st.header("📋 Filtered Data Table")
    min_date = clean_df["Date"].min().date()
    max_date = clean_df["Date"].max().date()
    start_date = pd.to_datetime(min_date)
    end_date = pd.to_datetime(max_date)
    filtered_df = clean_df[(clean_df["Date"] >= start_date) & (clean_df["Date"] <= end_date)]
    st.dataframe(filtered_df, use_container_width=True, height=400)

    st.header("📁 Raw AAII Sentiment Excel File")
    st.dataframe(raw_df)

# ---------------------------- TAB 2 ----------------------------------
with tab2:
    st.markdown("## :chart_with_upwards_trend: Interactive Dashboard")

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
# ---------------------------- TAB 7 ----------------------------------
# 📌 Tab 7: Deep Q-Learning Strategy
with tab7:
    st.markdown("""
    ### 🧠 Deep Q-Learning Strategy
    This strategy uses Deep Q-Learning to learn an optimal trading policy based on sentiment and price momentum.

    - **State:** z-scores of bullish sentiment, bearish sentiment, bull-bear spread, 4-week price return, volatility, moving average, and lagged return
    - **Actions:** -1 (short), 0 (neutral), 1 (long)
    - **Reward:** Next week's return * action (reward shaping applied)
    - **Training Range:** Selectable
    - **Testing Range:** Selectable
    """)

    df = clean_df.copy()

    # Training and testing sliders
    st.subheader("📆 Training and Testing Period")
    min_year = int(df["Date"].dt.year.min())
    max_year = int(df["Date"].dt.year.max())

    col1, col2 = st.columns(2)
    with col1:
        train_start = st.slider("Training Start Year", min_value=min_year, max_value=max_year - 2, value=2000)
    with col2:
        train_end = st.slider("Training End Year", min_value=train_start + 1, max_value=max_year - 1, value=2015)

    # Filter and engineer features
    df_ml = df.copy()
    df_ml['Bullish_z'] = zscore(df_ml['Bullish'])
    df_ml['Bearish_z'] = zscore(df_ml['Bearish'])
    df_ml['Spread_z'] = zscore(df_ml['Bullish'] - df_ml['Bearish'])
    df_ml['4w_return'] = df_ml['SP500_Close'].pct_change(4)
    df_ml['volatility'] = df_ml['SP500_Close'].pct_change().rolling(window=4).std()
    df_ml['ma'] = df_ml['SP500_Close'].rolling(window=4).mean()
    df_ml['lagged_return'] = df_ml['SP500_Close'].pct_change().shift(1)
    df_ml['Future_Return'] = df_ml['SP500_Close'].pct_change().shift(-1)
    df_ml.dropna(inplace=True)

    features = ['Bullish_z', 'Bearish_z', 'Spread_z', '4w_return', 'volatility', 'ma', 'lagged_return']

    # Define actions using thresholds
    df_ml['Action'] = np.where(df_ml['Future_Return'] > 0.002, 1,
                        np.where(df_ml['Future_Return'] < -0.002, -1, 0))

    # Split train and test sets
    df_train = df_ml[(df_ml['Date'].dt.year >= train_start) & (df_ml['Date'].dt.year <= train_end)]
    df_test = df_ml[df_ml['Date'].dt.year > train_end]

    X_train = df_train[features].values
    y_train = df_train['Action'].values
    X_test = df_test[features].values
    y_test = df_test['Action'].values

    # Ensure no NaNs or infs
    valid_train = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    valid_test = np.isfinite(X_test).all(axis=1) & np.isfinite(y_test)

    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_test, y_test = X_test[valid_test], y_test[valid_test]
    test_dates = df_test['Date'].values[valid_test]
    test_prices = df_test['SP500_Close'].values[valid_test]

    # Train neural network
    model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, random_state=42)
    model.fit(X_train, y_train)

    # Predict Q-values and determine action
    q_values = model.predict(X_test)
    actions_test = np.where(q_values > 0.25, 1, np.where(q_values < -0.25, -1, 0))

    # Display action distribution
    st.subheader("🧠 Training Action Distribution:")
    action_counts_train = {
        "0": int(np.sum(y_train == 0)),
        "1": int(np.sum(y_train == 1)),
        "-1": int(np.sum(y_train == -1))
    }
    st.write(action_counts_train)

    # Portfolio simulation
    portfolio_returns = df_test['Future_Return'].values[valid_test] * actions_test
    bh_returns = df_test['Future_Return'].values[valid_test]

    q_cum = pd.Series((1 + portfolio_returns).cumprod() * 10000, index=test_dates)
    bh_cum = pd.Series((1 + bh_returns).cumprod() * 10000, index=test_dates)

    df_plot = pd.DataFrame({
        'Date': test_dates,
        'Q_Learning': q_cum.values,
        'BuyHold': bh_cum.values
    })

    chart = alt.Chart(df_plot).transform_fold(["Q_Learning", "BuyHold"], as_=["Strategy", "Portfolio Value"]).mark_line().encode(
        x='Date:T',
        y=alt.Y('Portfolio Value:Q', title='Portfolio Value'),
        color='Strategy:N'
    ).properties(height=350)

    st.altair_chart(chart, use_container_width=True)

    # Performance Summary
    st.subheader("📊 Performance Summary")
    try:
        q_return = (q_cum[-1] / q_cum[0] - 1) * 100
        bh_return = (bh_cum[-1] / bh_cum[0] - 1) * 100
        st.markdown(f"""
        - **Deep Q-Learning Strategy Return:** {q_return:.2f}%  
        - **Buy & Hold Return:** {bh_return:.2f}%
        """)
    except Exception as e:
        st.warning(f"Could not compute return values: {e}")

    action_counts_test = {
        "Long": int(np.sum(actions_test == 1)),
        "Short": int(np.sum(actions_test == -1)),
        "Neutral": int(np.sum(actions_test == 0))
    }
    st.write("Action Distribution (Test Set):")
    st.write(action_counts_test)

# ---------------------------- TAB 8 ----------------------------------
with tab8:
    import plotly.graph_objects as go

    st.markdown("### 🔷 Fear & Greed Index")
    st.write("This indicator dynamically estimates current market sentiment based on AAII bullish/bearish sentiment and price momentum.")
    st.markdown("*The score is the average of two normalized components: the Bull-Bear sentiment spread and the 4-week return of the S&P 500.*")

    # Compute a simple dynamic fear & greed score (0–100 scale)
    df_fg = clean_df.copy()
    df_fg["BullBearSpread"] = df_fg["Bullish"] - df_fg["Bearish"]
    df_fg["Momentum"] = df_fg["SP500_Close"].pct_change(4)

    # Normalize both components to [0, 1] then scale to 100
    bb_scaled = (df_fg["BullBearSpread"] - df_fg["BullBearSpread"].min()) / (df_fg["BullBearSpread"].max() - df_fg["BullBearSpread"].min())
    mo_scaled = (df_fg["Momentum"] - df_fg["Momentum"].min()) / (df_fg["Momentum"].max() - df_fg["Momentum"].min())

    df_fg["FG_Score"] = ((bb_scaled + mo_scaled) / 2 * 100).clip(0, 100)
    df_fg = df_fg.dropna()

    # Latest score
    current_score = int(df_fg["FG_Score"].iloc[-1])

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_score,
        title={'text': "Fear & Greed Index"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 25], 'color': '#ffe6e6'},        # Extreme Fear
                {'range': [25, 50], 'color': '#fff5cc'},       # Fear
                {'range': [50, 75], 'color': '#e6ffe6'},       # Greed
                {'range': [75, 100], 'color': '#ccffcc'}       # Extreme Greed
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Sentiment label with explanation
    def get_sentiment_label(score):
        if score < 25:
            return "Extreme Fear"
        elif score < 50:
            return "Fear"
        elif score < 75:
            return "Greed"
        else:
            return "Extreme Greed"

    sentiment_label = get_sentiment_label(current_score)
    label_descriptions = {
        "Extreme Fear": "🔴 **Extreme Fear** – Investors are very worried.",
        "Fear": "🟠 **Fear** – Investors are cautious.",
        "Neutral": "🟡 **Neutral** – Market is balanced.",
        "Greed": "🟢 **Greed** – Investors are optimistic.",
        "Extreme Greed": "🟣 **Extreme Greed** – Investors are euphoric."
    }
    description = label_descriptions.get(sentiment_label, "")
    st.markdown(f"<h2 style='text-align: center;'>{description}</h2>", unsafe_allow_html=True)

    # Historical Sentiment Snapshots
    st.subheader("🕰️ Historical Sentiment Snapshots")
    dates = {
        "Previous Close": -1,
        "1 Week Ago": -5,
        "1 Month Ago": -21,
        "1 Year Ago": -52
    }

    cols = st.columns(len(dates))
    for i, (label, idx) in enumerate(dates.items()):
        val = int(df_fg["FG_Score"].iloc[idx])
        cols[i].metric(label, get_sentiment_label(val), val)

    # Last updated
    try:
        st.caption(f"Last updated {df_fg['Date'].iloc[-1].strftime('%B %d at %I:%M %p')} ET")
    except Exception:
        st.caption("Last updated: Unavailable")


st.set_page_config(page_title="CNN F&G replication", layout="wide")

# ----------------- TAB 9 ------------------------

# ------------------------- TAB 9: CNN Fear & Greed Replication -------------------------
with tab9:
    st.markdown("## 😱 Fear & Greed Index")
    st.markdown("""
    This tab replicates the CNN Fear & Greed Index using seven financial indicators from Yahoo Finance.  
    \nThe final score ranges from 0 (extreme fear) to 100 (extreme greed). Each indicator contributes equally.
    """)

    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=365)

    # Download Yahoo Finance data
    try:
        sp500 = yf.download("^GSPC", start=start, end=end)["Close"]
        vix = yf.download("^VIX", start=start, end=end)["Close"]
        hyg = yf.download("HYG", start=start, end=end)["Close"]
        spy = yf.download("SPY", start=start, end=end)["Close"]
        tlt = yf.download("TLT", start=start, end=end)["Close"]
    except Exception as e:
        st.error("❌ Failed to fetch data from Yahoo Finance.")
        st.exception(e)
        st.stop()

    if any(x is None or x.empty for x in [sp500, vix, hyg, spy, tlt]):
        st.error("One or more data series could not be downloaded properly.")
        st.stop()

    df = pd.concat([sp500, vix, hyg, spy, tlt], axis=1)
    df.columns = ["SP500", "VIX", "HYG", "SPY", "TLT"]
    df.dropna(inplace=True)

    # 1. Market Momentum: 125-day moving average vs current price
    ma_125 = df["SP500"].rolling(window=125).mean()
    momentum_score = 100 * (df["SP500"].iloc[-1] - ma_125.iloc[-1]) / ma_125.iloc[-1]

    # 2. Stock Price Strength: % above MA
    strength_score = 100 * np.mean(df["SP500"] > ma_125)

    # 3. Volatility: Inverted VIX (higher VIX = more fear)
    vix_z = (df["VIX"] - df["VIX"].mean()) / df["VIX"].std()
    vix_score = 100 - (vix_z.iloc[-1] * 20 + 50)

    # 4. Stock Price Breadth: Advance-decline proxy using HYG (risk appetite)
    hyg_ma = df["HYG"].rolling(window=30).mean()
    breadth_score = 100 * (df["HYG"].iloc[-1] - hyg_ma.iloc[-1]) / hyg_ma.iloc[-1]

    # 5. Put & Call Options: not available — proxy using VIX/SPY
    putcall_score = vix_score  # duplicated proxy

    # 6. Junk Bond Demand: HYG vs TLT
    bond_score = 100 * (df["HYG"].iloc[-1] / df["TLT"].iloc[-1]) - 50

    # 7. Safe Haven Demand: SPY vs TLT (inverse)
    safehaven_score = 100 * (df["SPY"].iloc[-1] / df["TLT"].iloc[-1]) - 50

    # Normalize and combine all 7 indicators equally
    components = [momentum_score, strength_score, vix_score,
                  breadth_score, putcall_score, bond_score, safehaven_score]

    normalized = [min(max(c, 0), 100) for c in components]
    final_score = int(np.mean(normalized))

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=final_score,
        title={'text': "Fear & Greed Index"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 25], 'color': '#ffcccc'},
                {'range': [25, 50], 'color': '#fff2cc'},
                {'range': [50, 75], 'color': '#d9f2d9'},
                {'range': [75, 100], 'color': '#b6d7a8'},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Label
    def fg_label(score):
        if score < 25:
            return "😱 Extreme Fear"
        elif score < 50:
            return "😨 Fear"
        elif score < 75:
            return "😐 Neutral"
        else:
            return "😄 Greed"

    st.subheader("📉 Market Sentiment Classification")
    st.markdown(f"**Current market mood:** `{fg_label(final_score)}` — Score: **{final_score}/100**")
