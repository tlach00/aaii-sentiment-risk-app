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

tab1, tab2, tab3, tab7, tab8, tab9 = st.tabs([
    "📁 Raw Excel Viewer",
    "📈 Interactive Dashboard",
    "🧪 Z-Score Strategy Backtest",
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



#with tab9:
    st.markdown("## 😱 Fear & Greed Index")
    st.markdown("""
    This tab replicates the CNN Fear & Greed Index using seven financial indicators from Yahoo Finance.

    - The final score ranges from 0 (extreme fear) to 100 (extreme greed).
    - Each indicator contributes equally and is normalized using **z-scores** (realistic scale).
    - Data is fetched from Yahoo Finance and covers 2007 to today.
    """)

    import yfinance as yf
    import pandas as pd
    import numpy as np
    import datetime
    import plotly.graph_objects as go
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import streamlit as st

    # Date range
    start = datetime.date(2007, 1, 1)
    end = datetime.date.today()

    # Add date range slider for FNG chart only
    fng_range = st.slider("Select date range for F&G Index chart:", min_value=start, max_value=end,
                          value=(start, end), format="YYYY-MM-DD")

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

        # Compute indicators
        momentum_ma = data["SP500"].rolling(125).mean()
        momentum = 100 * (data["SP500"] - momentum_ma) / momentum_ma

        strength = 100 * (data["SP500"] > momentum_ma).rolling(50).mean()

        spy_returns = data["SPY"].pct_change()
        breadth = 100 * spy_returns.rolling(20).mean()

        put_call = 100 - (data["VIX"].rolling(5).mean() - data["VIX"].mean()) / data["VIX"].std() * 20

        vix_ma = data["VIX"].rolling(50).mean()
        volatility = 100 - ((data["VIX"] - vix_ma) / vix_ma * 100)

        safe_haven = (data["SPY"] / data["TLT"]).pct_change().rolling(20).mean() * 100
        junk_demand = (data["HYG"] / data["LQD"]).pct_change().rolling(20).mean() * 100

        def normalize(series):
            z = (series - series.mean()) / series.std()
            return 50 + z * 10

        fng_df = pd.DataFrame({
            "momentum": normalize(momentum),
            "strength": normalize(strength),
            "breadth": normalize(breadth),
            "putcall": normalize(put_call),
            "volatility": normalize(volatility),
            "safehaven": normalize(safe_haven),
            "junk": normalize(junk_demand),
        })

        fng_df["FNG_Index"] = fng_df.mean(axis=1)
        fng_df.dropna(inplace=True)

        fng_filtered = fng_df.loc[fng_range[0]:fng_range[1]]

        # Current value
        latest_score = int(fng_df["FNG_Index"].iloc[-1])
        latest_date = fng_df.index[-1].strftime("%B %d, %Y")

        def fg_label(score):
            if score < 25:
                return "😱 Extreme Fear"
            elif score < 50:
                return "😨 Fear"
            elif score < 75:
                return "😐 Neutral"
            else:
                return "😄 Greed"

        # Gauge chart
        st.plotly_chart(go.Figure(go.Indicator(
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
                    {'range': [75, 100], 'color': '#b6d7a8'}
                ]
            })), use_container_width=True)

        st.markdown(f"**Current market mood on {latest_date}:** {fg_label(latest_score)} — Score: **{latest_score}/100**")

        # Historical line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fng_filtered.index, y=fng_filtered["FNG_Index"], mode="lines", name="F&G Index", line=dict(color='steelblue')))
        fig.add_shape(type="rect", x0=fng_filtered.index[0], x1=fng_filtered.index[-1], y0=0, y1=25, fillcolor="#ffcccc", opacity=0.3, line_width=0)
        fig.add_shape(type="rect", x0=fng_filtered.index[0], x1=fng_filtered.index[-1], y0=75, y1=100, fillcolor="#d9f2d9", opacity=0.3, line_width=0)
        fig.update_layout(yaxis_title='Index Value (0–100)', xaxis_title='Date', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # ---------------- Machine Learning Strategy -------------------
        st.subheader("🧰 ML Strategy using F&G Components")
        features = fng_df.copy()
        features["target"] = data["SPY"].pct_change().shift(-1)
        features.dropna(inplace=True)
        features["label"] = np.where(features["target"] > 0.001, 1, np.where(features["target"] < -0.001, -1, 0))

        split_date = "2020-01-01"
        X_train = features.loc[:split_date].drop(columns=["target", "label"])
        y_train = features.loc[:split_date]["label"]
        X_test = features.loc[split_date:].drop(columns=["target", "label"])
        y_test = features.loc[split_date:]["label"]
        returns_test = features.loc[split_date:]["target"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(multi_class='ovr', max_iter=1000)
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

        capital = 10000
        strat_values = [capital]
        bh_values = [capital]
        positions = []
        daily_returns = []

        for i, r in enumerate(returns_test):
            a = predictions[i]
            daily_r = r * a
            positions.append(a)
            daily_returns.append(daily_r)
            strat_values.append(strat_values[-1] * (1 + daily_r))
            bh_values.append(bh_values[-1] * (1 + r))

        ml_df = pd.DataFrame({
            "Date": returns_test.index,
            "Position": positions,
            "Daily Return": daily_returns,
            "Strategy Value": strat_values[1:],
            "BuyHold": bh_values[1:]
        })

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ml_df["Date"], y=ml_df["Strategy Value"], name="ML Strategy"))
        fig2.add_trace(go.Scatter(x=ml_df["Date"], y=ml_df["BuyHold"], name="Buy & Hold", line=dict(dash="dash", color="black")))
        fig2.update_layout(title="ML Strategy vs Buy & Hold", xaxis_title="Date", yaxis_title="Portfolio Value", height=500)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 📈 Trade Log")
        position_labels = {1: "Long", 0: "Neutral", -1: "Short"}
        ml_df["Label"] = [position_labels[p] for p in ml_df["Position"]]
        st.dataframe(ml_df[["Date", "Label", "Daily Return", "Strategy Value"]].set_index("Date"))

    except Exception as e:
        st.error("Error computing indicators")
        st.exception(e)
