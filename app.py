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

## datas from AAII survey
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

## function for the F&G cnn replica
@st.cache_data
def load_fng_data():
    tickers = ["^GSPC", "^VIX", "SPY", "TLT", "HYG", "LQD"]
    data = yf.download(tickers, start="2007-01-01")["Close"]
    data.dropna(inplace=True)

    def normalize(series):
        z = (series - series.mean()) / series.std()
        return np.clip(50 + z * 25, 0, 100)

    # Indicators
    momentum_ma = data["^GSPC"].rolling(window=125).mean()
    momentum = 100 * (data["^GSPC"] - momentum_ma) / momentum_ma
    strength = 100 * (data["^GSPC"] > momentum_ma).rolling(window=50).mean()
    spy_returns = data["SPY"].pct_change()
    breadth = 100 * spy_returns.rolling(20).mean()
    put_call = 100 - (data["^VIX"].rolling(5).mean() - data["^VIX"].mean()) / data["^VIX"].std() * 20
    vix_ma = data["^VIX"].rolling(window=50).mean()
    volatility = 100 - ((data["^VIX"] - vix_ma) / vix_ma * 100)
    safe_haven = (data["SPY"] / data["TLT"]).pct_change().rolling(20).mean() * 100
    junk_demand = (data["HYG"] / data["LQD"]).pct_change().rolling(20).mean() * 100

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
    return fng_df, data

# ✅ Call the F&G function here and store results globally
fng_df, data = load_fng_data()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Raw Excel Viewer",
    "📈 AAII Sentiment survey",
    "😱 CNN F&G replication", 
    "👻 Stock F&G", 
    "📟 F&G in Risk Management"
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


# ---------------------------- TAB 3 ----------------------------
with tab3:
    import plotly.graph_objects as go

    st.markdown("## 🧐 CNN-Style Fear & Greed Replication")

    # ✅ Use globally loaded fng_df (no need to call load_fng_data again)

    with st.expander("🧠 How This CNN-Style Fear & Greed Index Works"):
        st.markdown("""
        The official **CNN Fear & Greed Index** includes:

        1. **Momentum** – S&P 500 vs. 125-day MA  
        2. **Strength** – % of days above MA  
        3. **Breadth** – 20-day SPY returns  
        4. **Put/Call** – VIX-based proxy  
        5. **Volatility** – VIX vs. 50-day MA  
        6. **Safe Haven** – SPY/TLT returns  
        7. **Junk Demand** – HYG/LQD returns

        All indicators are normalized (Z-score scaled to 0–100), then averaged.
        """)

    col1, col2 = st.columns(2)

    with col1:
        latest_score = int(fng_df["FNG_Index"].iloc[-1])
        fig = go.Figure(go.Indicator(
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
        st.plotly_chart(fig, use_container_width=True)

        def get_sentiment_label(score):
            if score < 25: return "🔴 **Extreme Fear**"
            elif score < 50: return "🟠 **Fear**"
            elif score < 60: return "🟡 **Neutral**"
            elif score < 75: return "🟢 **Greed**"
            else: return "🟣 **Extreme Greed**"

        st.markdown(f"<h3 style='text-align: center;'>{get_sentiment_label(latest_score)}</h3>", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🕰️ Sentiment Snapshots")

        def sentiment_tag(score):
            if score < 25: return "Extreme Fear", "#ffcccc"
            elif score < 50: return "Fear", "#ffe6cc"
            elif score < 60: return "Neutral", "#dddddd"
            elif score < 75: return "Greed", "#ccffcc"
            else: return "Extreme Greed", "#aaffaa"

        snapshots = {
            "Previous Close": -2,
            "1 Week Ago": -5,
            "1 Month Ago": -21,
            "1 Year Ago": -252
        }

        for label, idx in snapshots.items():
            try:
                score = int(fng_df["FNG_Index"].iloc[idx])
                sentiment, color = sentiment_tag(score)
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px dashed #ccc;'>
                        <div style='font-size: 14px;'>{label}</div>
                        <div style='font-weight: bold;'>{sentiment}</div>
                        <div style='background-color: {color}; border-radius: 50%; padding: 6px 12px; font-weight: bold;'>{score}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except:
                st.markdown(f"<i>Data unavailable for {label}</i>", unsafe_allow_html=True)

    st.markdown("### 📉 Historical Fear & Greed Index")
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
# ---------------- tab 4 ----------------
with tab4:
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

    # Optional: manual refresh
    # if st.button("🔄 Refresh Top 10 Gauges"):
    #     st.cache_data.clear()

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

            st.markdown("#### Historical Fear & Greed Index")
            fig_fg = px.line(fng_range, title="F&G Index", labels={"value": "F&G Score", "index": "Date"})
            fig_fg.update_layout(height=300)
            st.plotly_chart(fig_fg, use_container_width=True, key="line_fg")

            st.markdown("#### Stock Price History")
            price_data = yf.download(selected, start=date_range[0], end=date_range[1])["Close"]
            fig_price = px.line(price_data, title="Price", labels={"value": "Price", "index": "Date"})
            fig_price.update_layout(height=300)
            st.plotly_chart(fig_price, use_container_width=True, key="line_price")
        else:
            st.warning("Could not retrieve or compute data for this ticker.")

# ---------------------------- TAB 5 ----------------------------------
with tab5:
    st.markdown("## 🔐 Risk Management Overlay using Fear & Greed")

    try:
        # Load F&G and price data
        fng_df, data = load_fng_data()

        latest = fng_df.iloc[-1]
        last_date = latest.name.date()

        st.subheader("📅 Date")
        st.write(f"{last_date}")

        # Calculate 1-day SPY returns
        spy_returns = data["SPY"].pct_change().dropna()
        fng_df = fng_df.loc[spy_returns.index]

        # Historical VaR and CVaR over full sample
        var_95 = np.percentile(spy_returns, 5)
        cvar_95 = spy_returns[spy_returns <= var_95].mean()

        st.metric("📉 1-Day VaR (95%)", f"{var_95 * 100:.2f}%")
        st.metric("📉 1-Day CVaR (95%)", f"{cvar_95 * 100:.2f}%")

        # Show F&G value
        score = latest["FNG_Index"]
        st.metric("🧠 F&G Score", f"{score:.1f}")

        def get_sentiment_label(score):
            if score < 25:
                return "🔴 Extreme Fear"
            elif score < 50:
                return "🟠 Fear"
            elif score < 60:
                return "🟡 Neutral"
            elif score < 75:
                return "🟢 Greed"
            else:
                return "🟣 Extreme Greed"

        st.markdown(f"### Current Sentiment Regime: **{get_sentiment_label(score)}**")

        # -------------------- GRAPH SECTION ----------------------
        st.markdown("### 📊 VaR & CVaR Overlay with Fear & Greed")

        # Rolling metrics
        window = 100
        confidence = 0.95
        z = np.abs(np.percentile(np.random.randn(100000), (1 - confidence) * 100))

        fng_df["VaR"] = -spy_returns.rolling(window).std() * z * 100
        fng_df["CVaR"] = spy_returns.rolling(window).apply(
            lambda x: -x[x <= np.percentile(x, (1 - confidence) * 100)].mean(), raw=True
        ) * 100

        fig_overlay = go.Figure()

        # VaR and CVaR
        fig_overlay.add_trace(go.Scatter(
            x=fng_df.index, y=fng_df["VaR"], name="VaR (95%)", yaxis="y1", line=dict(color="blue")
        ))
        fig_overlay.add_trace(go.Scatter(
            x=fng_df.index, y=fng_df["CVaR"], name="CVaR (95%)", yaxis="y1", line=dict(color="red", dash="dot")
        ))

        # FNG Index
        fig_overlay.add_trace(go.Scatter(
            x=fng_df.index, y=fng_df["FNG_Index"], name="F&G Index", yaxis="y2", line=dict(color="green")
        ))

        # Layout
        fig_overlay.update_layout(
            title="1-Day VaR & CVaR vs Fear & Greed Index",
            xaxis=dict(title="Date"),
            yaxis=dict(
                title="VaR / CVaR (%)",
                titlefont=dict(color="blue"),
                tickfont=dict(color="blue"),
                side="left"
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
            height=500,
            margin=dict(l=40, r=40, t=40, b=30)
        )

        # Sentiment thresholds
        fig_overlay.add_shape(type="line", x0=fng_df.index[0], x1=fng_df.index[-1], y0=25, y1=25, yref="y2",
                              line=dict(color="gray", dash="dash"))
        fig_overlay.add_shape(type="line", x0=fng_df.index[0], x1=fng_df.index[-1], y0=75, y1=75, yref="y2",
                              line=dict(color="gray", dash="dash"))

        st.plotly_chart(fig_overlay, use_container_width=True)

    except Exception as e:
        st.error("⚠️ Could not compute risk overlay.")
        st.exception(e)
