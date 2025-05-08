import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import zscore
import warnings
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from tqdm import tqdm
from datetime import datetime, timedelta
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
tab1, tab2, tab3, tab7, tab9, tab10 = st.tabs([
    "📁 Raw Excel Viewer",
    "📈 Interactive Dashboard",
    "🧪 Z-Score Strategy Backtest",
    "🧠 Deep Q-Learning Strategy",
    "😱 CNN F&G replication", 
    "👻 F&G mini-gauge dashboard"
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
    st.markdown("## :chart_with_upwards_trend: AAII Investor sentiment survey")

    # AAII summary block
    st.markdown("""
    ### 🧠 About the AAII Sentiment Survey
    The **AAII Investor Sentiment Survey** is a weekly gauge of market expectations among individual investors.  
    It asks a single question:  
    **"Do you feel the direction of the stock market over the next six months will be up (bullish), no change (neutral), or down (bearish)?"**

    - **Frequency**: Weekly (runs from Thursday to Wednesday)
    - **Participants**: AAII members (mostly individual investors)
    - **Relevance**: Often used as a **contrarian indicator** — high bullishness can signal potential tops, and high bearishness can signal bottoms.
    - **Long-term averages** (since 1987):  
      - 🐂 **Bullish**: ~37.5%  
      - ≡ **Neutral**: ~31.5%  
      - 🐻 **Bearish**: ~31.0%  
    """)

    # === AAII Investor sentiment Index  ===
    import plotly.graph_objects as go
    st.markdown("### 🔷 AAII Investor sentiment Index")
    st.write("This indicator dynamically estimates current market sentiment based on AAII bullish/bearish sentiment and price momentum.")
    st.markdown("*The score is the average of two normalized components: the Bull-Bear sentiment spread and the 4-week return of the S&P 500.*")

    df_fg = clean_df.copy()
    df_fg["BullBearSpread"] = df_fg["Bullish"] - df_fg["Bearish"]
    df_fg["Momentum"] = df_fg["SP500_Close"].pct_change(4)

    bb_scaled = (df_fg["BullBearSpread"] - df_fg["BullBearSpread"].min()) / (df_fg["BullBearSpread"].max() - df_fg["BullBearSpread"].min())
    mo_scaled = (df_fg["Momentum"] - df_fg["Momentum"].min()) / (df_fg["Momentum"].max() - df_fg["Momentum"].min())
    df_fg["FG_Score"] = ((bb_scaled + mo_scaled) / 2 * 100).clip(0, 100)
    df_fg.dropna(inplace=True)

    current_score = int(df_fg["FG_Score"].iloc[-1])

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_score,
        title={'text': "Fear & Greed Index"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 25], 'color': '#ffe6e6'},
                {'range': [25, 50], 'color': '#fff5cc'},
                {'range': [50, 75], 'color': '#e6ffe6'},
                {'range': [75, 100], 'color': '#ccffcc'}
            ]
        }
    ))
    st.plotly_chart(go.Figure(fig), use_container_width=True)

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

    st.subheader("🕰️ Historical Sentiment Snapshots")
    dates = {
        "Previous Close": -1,
        "1 Week Ago": -5,
        "1 Month Ago": -21,
        "1 Year Ago": -252
    }
    cols = st.columns(len(dates))
    for i, (label, idx) in enumerate(dates.items()):
        val = int(df_fg["FG_Score"].iloc[idx])
        cols[i].metric(label, get_sentiment_label(val), val)

    try:
        st.caption(f"Last updated {df_fg['Date'].iloc[-1].strftime('%B %d at %I:%M %p')} ET")
    except Exception:
        st.caption("Last updated: Unavailable")
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
    st.markdown("### :newspaper: S&P 500 Weekly Close")
    chart1 = alt.Chart(filtered_df).mark_line(color='black').encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('SP500_Close:Q', title='Price')
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

# ------------------------- TAB 9: CNN Fear & Greed Replication + ML Strategy -------------------------
with tab9:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import datetime
    import plotly.graph_objects as go
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import plotly.express as px

    st.markdown("## 🧐 CNN-Style Fear & Greed Replication + ML Strategy")

    # Date range up to today
    end = datetime.datetime.today()
    start = datetime.datetime(2007, 1, 1)

    tickers = {
        "SP500": "^GSPC",
        "VIX": "^VIX",
        "SPY": "SPY",
        "TLT": "TLT",
        "HYG": "HYG",
        "LQD": "LQD"
    }

    try:
        data = yf.download(list(tickers.values()), start=start, end=end)["Close"]
        data.columns = list(tickers.keys())
        data.dropna(inplace=True)

        # Indicators
        momentum_ma = data["SP500"].rolling(window=125).mean()
        momentum = 100 * (data["SP500"] - momentum_ma) / momentum_ma
        strength = 100 * (data["SP500"] > momentum_ma).rolling(window=50).mean()
        spy_returns = data["SPY"].pct_change()
        breadth = 100 * spy_returns.rolling(20).mean()
        put_call = 100 - (data["VIX"].rolling(5).mean() - data["VIX"].mean()) / data["VIX"].std() * 20
        vix_ma = data["VIX"].rolling(window=50).mean()
        volatility = 100 - ((data["VIX"] - vix_ma) / vix_ma * 100)
        safe_haven = (data["SPY"] / data["TLT"]).pct_change().rolling(20).mean() * 100
        junk_demand = (data["HYG"] / data["LQD"]).pct_change().rolling(20).mean() * 100

        # CNN Z-score scaling
        def normalize(series):
            z = (series - series.mean()) / series.std()
            return np.clip(50 + z * 25, 0, 100)

        fng_df = pd.DataFrame({
            "momentum": normalize(momentum),
            "strength": normalize(strength),
            "breadth": normalize(breadth),
            "putcall": normalize(put_call),
            "volatility": normalize(volatility),
            "safehaven": normalize(safe_haven),
            "junk": normalize(junk_demand),
        }, index=data.index)

        fng_df["FNG_Index"] = fng_df.mean(axis=1)
        fng_df["FNG_Smooth"] = fng_df["FNG_Index"].rolling(window=100).mean()
        fng_df.dropna(inplace=True)

        with st.expander("🧠 How This CNN-Style Fear & Greed Index Works"):
            st.markdown("""
            The official **CNN Fear & Greed Index** measures market sentiment using **7 equally weighted indicators**, each scaled from 0 to 100 and averaged:
        
            1. **Stock Price Momentum** – S&P 500 vs. 125-day moving average  
            2. **Stock Price Strength** – 52-week highs vs. lows (NYSE)  
            3. **Stock Price Breadth** – Advancing vs. declining volume  
            4. **Put/Call Options** – Ratio of bearish to bullish bets  
            5. **Market Volatility** – VIX vs. its 50-day average  
            6. **Safe Haven Demand** – Stock vs. Treasury performance  
            7. **Junk Bond Demand** – Risk appetite from junk vs. investment-grade bonds
        
            Our replication mimics this index using **free data from Yahoo Finance**:
        
            - 🟢 **Momentum**: S&P 500 vs. 125-day MA (official method)  
            - 🟢 **Strength**: % of days above MA (proxy for 52W highs)  
            - 🟡 **Breadth**: 20-day SPY returns (proxy for volume flows)  
            - 🟡 **Put/Call**: VIX z-score proxy  
            - 🟢 **Volatility**: VIX vs. 50-day MA (official method)  
            - 🟢 **Safe Haven**: SPY/TLT 20-day return  
            - 🟢 **Junk Demand**: HYG/LQD 20-day return
        
            Each indicator is **z-scored and scaled to 0–100**, then averaged to form the final score.  
            This allows tracking investor sentiment in a transparent, replicable way.
            """)
        # Gauge
        latest_score = int(fng_df["FNG_Index"].iloc[-1])
        latest_date = fng_df.index[-1].strftime("%B %d, %Y")

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_score,
            title={'text': "Fear & Greed Index"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 25], 'color': '#ffcccc'},
                    {'range': [25, 50], 'color': '#fff2cc'},
                    {'range': [50, 75], 'color': '#d9f2d9'},
                    {'range': [75, 100], 'color': '#b6d7a8'},
                ]
            }
        ))
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Historical chart
        st.markdown("### 📉 Historical Fear & Greed Index (Since 2007)")
        fig_fng = go.Figure()
        fig_fng.add_trace(go.Scatter(x=fng_df.index, y=fng_df["FNG_Index"], name="F&G Index", mode="lines"))
        fig_fng.add_trace(go.Scatter(x=fng_df.index, y=fng_df["FNG_Smooth"], name="100-day MA", mode="lines", line=dict(color="red")))
        fig_fng.update_layout(
            shapes=[
                dict(type="rect", xref="x", yref="y", x0=fng_df.index[0], x1=fng_df.index[-1], y0=0, y1=25,
                     fillcolor="#ffcccc", opacity=0.2, line_width=0),
                dict(type="rect", xref="x", yref="y", x0=fng_df.index[0], x1=fng_df.index[-1], y0=75, y1=100,
                     fillcolor="#d9f2d9", opacity=0.2, line_width=0),
            ],
            yaxis_title="Index Value (0–100)",
            xaxis_title="Date",
            height=600,
            margin=dict(l=40, r=40, t=30, b=30)
        )
        st.plotly_chart(fig_fng, use_container_width=True)

        # ML Strategy below chart
        st.markdown("### 📈 ML Strategy vs Buy & Hold")

        ml_df = fng_df.copy()
        ml_df["target"] = data["SPY"].pct_change().shift(-1)
        ml_df.dropna(inplace=True)
        ml_df["label"] = np.where(ml_df["target"] > 0.001, 1, np.where(ml_df["target"] < -0.001, -1, 0))

        split_date = "2020-01-01"
        X_train = ml_df.loc[:split_date].drop(columns=["target", "label"])
        y_train = ml_df.loc[:split_date]["label"]
        X_test = ml_df.loc[split_date:].drop(columns=["target", "label"])
        returns_test = ml_df.loc[split_date:]["target"]
        dates_test = returns_test.index

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

        capital = 10000
        strat = [capital]
        buy_hold = [capital]
        positions = []
        trades = []

        for i, r in enumerate(returns_test):
            action = predictions[i]  # -1, 0, 1
            positions.append(action)
            strat.append(strat[-1] * (1 + r * action))
            buy_hold.append(buy_hold[-1] * (1 + r))
            if i > 0 and predictions[i] != predictions[i-1]:
                trades.append({"Date": dates_test[i], "Position": action})

        strat = strat[1:]
        buy_hold = buy_hold[1:]

        plot_df = pd.DataFrame({
            "Date": dates_test,
            "ML Strategy": strat,
            "Buy & Hold": buy_hold
        })

        fig_perf = px.line(plot_df, x="Date", y=["ML Strategy", "Buy & Hold"],
                           labels={"value": "Portfolio Value", "variable": "Strategy"},
                           title="ML Strategy vs Buy & Hold")
        fig_perf.update_layout(height=400)
        st.plotly_chart(fig_perf, use_container_width=True)

        # Trade list
        st.markdown("### 📋 List of Trades (Strategy Position Changes)")
        trade_df = pd.DataFrame(trades)
        trade_df["Position"] = trade_df["Position"].map({1: "Long", 0: "Neutral", -1: "Short"})
        st.dataframe(trade_df.set_index("Date"))

    except Exception as e:
        st.error("❌ Error fetching or processing data.")
        st.exception(e)

# ---------------- tab 10 ----------------
with tab10:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    import datetime
    import streamlit as st

    st.markdown("## 🧬 Stock-Specific Fear & Greed Index Dashboard")

    with st.expander("❓ How is this index calculated?"):
        st.markdown("""
        The **Stock-Specific Fear & Greed Index** combines 6 key components:

        - **Volatility**: Stock’s 20-day and 50-day historical volatility.
        - **Safe Haven Demand**: SPY/TLT ratio.
        - **Junk Bond Demand**: HYG vs LQD yield spread.
        - **Sentiment**: Options skew (put vs call interest).
        - **Momentum**: Stock vs 125-day MA.
        - **Breadth**: 20-day return or RSI.

        Each feature is standardized (z-score), averaged, and scaled from 0 to 100.
        """)

    def compute_fg_score(ticker):
        try:
            end = datetime.datetime.today()
            start = end - datetime.timedelta(days=365)
            stock = yf.download(ticker, start=start, end=end)["Close"]
            market = yf.download(["SPY", "TLT", "HYG", "LQD"], start=start, end=end)["Close"]
            
            df = pd.DataFrame(index=stock.index)
            df["price"] = stock
            df["ret"] = stock.pct_change()
            df["vol_20"] = stock.pct_change().rolling(20).std()
            df["vol_50"] = stock.pct_change().rolling(50).std()
            df["vol"] = df[["vol_20", "vol_50"]].mean(axis=1)
            df["momentum"] = 100 * (stock - stock.rolling(125).mean()) / stock.rolling(125).mean()
            df["breadth"] = stock.pct_change(20)
            df["safehaven"] = (market["SPY"] / market["TLT"]).pct_change().rolling(20).mean()
            df["junk"] = (market["HYG"] / market["LQD"]).pct_change().rolling(20).mean()
            df.dropna(inplace=True)

            z = (df[["vol", "momentum", "breadth", "safehaven", "junk"]] - df[["vol", "momentum", "breadth", "safehaven", "junk"]].mean()) / df[["vol", "momentum", "breadth", "safehaven", "junk"]].std()
            fng_scaled = 50 + z.mean(axis=1) * 10
            return fng_scaled.clip(0, 100)
        except Exception as e:
            return None

    def plot_fng_gauge(ticker, score, key):
        color = "#d9f2d9" if score > 75 else "#e6f2cc" if score > 50 else "#fff2cc" if score > 25 else "#ffcccc"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 25], 'color': '#ffcccc'},
                    {'range': [25, 50], 'color': '#fff2cc'},
                    {'range': [50, 75], 'color': '#d9f2d9'},
                    {'range': [75, 100], 'color': '#b6d7a8'},
                ]
            }
        ))
        fig.update_layout(width=260, height=200, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=False, key=key)
        st.markdown(f"<div style='text-align:center; color:{color}; font-weight:bold;'>{ticker}</div>", unsafe_allow_html=True)

    @st.cache_data
    def load_top_gauges():
        results = {}
        for ticker in ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "TSLA", "UNH"]:
            fng_today = compute_fg_score(ticker)
            if fng_today is not None and not fng_today.empty:
                score = round(fng_today[-1], 1)
                results[ticker] = score
        return results

    st.markdown("### 🏆 Fear & Greed Today — Top S&P 500 Companies")
    cols = st.columns(4)
    top_scores = load_top_gauges()
    for i, (ticker, score) in enumerate(top_scores.items()):
        with cols[i % 4]:
            plot_fng_gauge(ticker, score, key=f"gauge_{ticker}")

    st.markdown("---")
    st.markdown("### 🔍 Explore Any S&P 500 Stock")

    with open("tickers_sp500.txt") as f:
        all_tickers = f.read().splitlines()

    selected = st.selectbox("Select a stock:", options=all_tickers)
    date_range = st.date_input("Select date range:", value=(datetime.date(2023, 1, 1), datetime.date.today()))

    if selected:
        fng_full = compute_fg_score(selected)
        if fng_full is not None and not fng_full.empty:
            fng_range = fng_full.loc[str(date_range[0]):str(date_range[1])]
            st.metric(f"Today's {selected} Fear & Greed Score", f"{round(fng_range[-1],1)}")

            st.markdown("#### 📈 Stock Price & Fear & Greed Index")

            # Download and align price data
            price_data = yf.download(selected, start=date_range[0], end=date_range[1])["Close"]
            merged = pd.concat([price_data.rename("Price"), fng_full.rename("FNG")], axis=1, join="inner")
            merged = merged.loc[str(date_range[0]):str(date_range[1])].dropna()

            fig_combined = go.Figure()

            # Add shaded regions for Extreme Fear and Extreme Greed
            fig_combined.add_shape(type="rect", xref="x", yref="y2",
                x0=merged.index[0], x1=merged.index[-1], y0=0, y1=25,
                fillcolor="#ffcccc", opacity=0.2, line_width=0)

            fig_combined.add_shape(type="rect", xref="x", yref="y2",
                x0=merged.index[0], x1=merged.index[-1], y0=75, y1=100,
                fillcolor="#d9f2d9", opacity=0.2, line_width=0)

            # Add price line (left y-axis)
            fig_combined.add_trace(go.Scatter(
                x=merged.index,
                y=merged["Price"],
                name="Price",
                yaxis="y1",
                line=dict(color="black")
            ))

            # Add F&G index line (right y-axis)
            fig_combined.add_trace(go.Scatter(
                x=merged.index,
                y=merged["FNG"],
                name="Fear & Greed Index",
                yaxis="y2",
                line=dict(color="green", dash="dot")
            ))

            fig_combined.update_layout(
                title=f"{selected}: Price and Fear & Greed Index",
                xaxis=dict(title="Date"),
                yaxis=dict(
                    title="Price",
                    titlefont=dict(color="black"),
                    tickfont=dict(color="black")
                ),
                yaxis2=dict(
                    title="F&G Index (0–100)",
                    titlefont=dict(color="green"),
                    tickfont=dict(color="green"),
                    overlaying="y",
                    side="right",
                    range=[0, 100]
                ),
                legend=dict(x=0.01, y=0.99),
                height=400,
                margin=dict(l=40, r=40, t=30, b=30)
            )

            st.plotly_chart(fig_combined, use_container_width=True)

        else:
            st.warning("Could not retrieve or compute data for this ticker.")
