import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import zscore
import warnings
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
st.set_page_config(page_title="AAII Sentiment & S&P 500 Risk Management Dashboard", layout="wide")
st.title(":bar_chart: AAII Sentiment & S&P 500 Risk Management Dashboard")

## AAII survey data
@st.cache_data
def load_raw_excel():
    return pd.read_excel("sentiment_data.xlsx", header=None)

def load_clean_data():
    df = pd.read_excel("sentiment_data.xlsx", skiprows=7, usecols="A:D,M", header=None)
    df.columns = ["Date", "Bullish", "Neutral", "Bearish", "SP500_Close"]
    df = df.dropna(subset=["Date", "Bullish", "Neutral", "Bearish", "SP500_Close"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    df["SP500_Return"] = df["SP500_Close"].pct_change() * 100

    # Ensure all sentiment percentages are scaled properly
    for col in ["Bullish", "Neutral", "Bearish"]:
        if df[col].max() <= 1:
            df[col] *= 100

    return df.dropna()

raw_df = load_raw_excel()
clean_df = load_clean_data()

## F&G Index function
@st.cache_data
def load_fng_data():
    tickers = ["^GSPC", "^VIX", "SPY", "TLT", "HYG", "LQD"]
    data = yf.download(tickers, start="2007-01-01")["Close"].dropna()
    data.index = pd.to_datetime(data.index)  # ✅ Ensure consistent datetime index

    def normalize(series):
        z = (series - series.mean()) / series.std()
        return np.clip(50 + z * 25, 0, 100)

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

# ✅ Load data globally for use in all tabs
fng_df, data = load_fng_data()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📓Read me",
    "📁 Raw Excel Viewer",
    "📈 AAII Sentiment survey",
    "😱 CNN F&G replication", 
    "👻 Stock F&G", 
    "📟 F&G in Risk Management",
    "⚖️ Dynamic Exposure Scaling & Stop-Loss Triggers",
    "🧨 F&G Stop-Loss",
    "⚙️ Dynamic stop loss Portfolio"
])
# ---------------------------- TAB 1 ----------------------------------
with tab1:
    st.markdown("## 🧭 App Introduction")

    st.markdown("""
    Over the past decade, markets have become increasingly sentiment-driven — shaped not only by fundamentals but by investor psychology, fear, greed, and narrative shifts.  
    Volatility spikes, meme stock frenzies, and algorithmic reactivity have highlighted how emotions can overpower logic in the short term.

    This dashboard was built to help bridge that gap between **traditional risk metrics** and **real-world investor behavior**.

    **Specifically, the app integrates:**
    - The **AAII Sentiment Survey** — a contrarian measure of retail investor mood
    - A **CNN-style Fear & Greed Index replica**
    - Tools to translate sentiment signals into **risk management frameworks**, including:
      - Dynamic exposure scaling  
      - VaR & CVaR adjustment  
      - Stop-loss triggers

    By combining these behavioral indicators with market data, the goal is to help you better understand:
    - When risk may be **mispriced**
    - When to **scale in or out** of exposure
    - And how to build **adaptive strategies** that respond not just to volatility, but to **investor psychology**

    Whether you're a portfolio manager, a quant, or simply a data-driven investor, this dashboard offers an interactive sandbox to explore how sentiment shapes performance.
    """)

# ---------------------------- TAB 2 ----------------------------------
with tab2:
    st.header("📋 Filtered Data Table")
    min_date = clean_df["Date"].min().date()
    max_date = clean_df["Date"].max().date()
    start_date = pd.to_datetime(min_date)
    end_date = pd.to_datetime(max_date)
    filtered_df = clean_df[(clean_df["Date"] >= start_date) & (clean_df["Date"] <= end_date)]
    st.dataframe(filtered_df, use_container_width=True, height=400)

    st.header("📁 Raw AAII Sentiment Excel File")
    st.dataframe(raw_df)

    st.header("📊 Historical Data Behind CNN Fear & Greed Index")
    st.dataframe(fng_df.tail(300), use_container_width=True, height=400)


# ---------------------------- TAB 3 ----------------------------------
with tab3:
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

    # === Bullish Sentiment Gauge Only ===
    st.markdown("### 🔷 AAII Bullish Sentiment Gauge")

    latest_bullish = clean_df["Bullish"].iloc[-1]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_bullish,
        title={'text': "Bullish Sentiment (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 25], 'color': '#ffe6e6'},
                {'range': [25, 50], 'color': '#fff5cc'},
                {'range': [50, 75], 'color': '#e6ffe6'},
                {'range': [75, 100], 'color': '#ccffcc'}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🕰️ Historical Sentiment Snapshots")
    dates = {
        "Previous Close": -1,
        "1 Week Ago": -5,
        "1 Month Ago": -21,
        "1 Year Ago": -252
    }
    cols = st.columns(len(dates))
    for i, (label, idx) in enumerate(dates.items()):
        val = clean_df["Bullish"].iloc[idx]
        cols[i].metric(label, f"{val:.1f}%")

    try:
        st.caption(f"Last updated {clean_df['Date'].iloc[-1].strftime('%B %d at %I:%M %p')} ET")
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

    st.markdown("### 🧠 Investor Sentiment Distribution")
    stacked_data = filtered_df.melt(
        id_vars=["Date"],
        value_vars=["Bearish", "Neutral", "Bullish"],
        var_name="Sentiment", value_name="Percentage"
    )

    # enforce stacking order by setting category dtype
    stacked_data["Sentiment"] = pd.Categorical(
        stacked_data["Sentiment"],
        categories=["Bearish", "Neutral", "Bullish"],
        ordered=True
    )

    sentiment_order = ["Bearish", "Neutral", "Bullish"]
    color_order = ["#f4aaaa", "#dddddd", "#c4e3cc"]

    base = alt.Chart(stacked_data).mark_bar().encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Percentage:Q', stack='normalize', title="Proportion (%)"),
        color=alt.Color('Sentiment:N', scale=alt.Scale(domain=sentiment_order, range=color_order))
    ).properties(height=300)
    st.altair_chart(base, use_container_width=True)

    st.markdown("### :chart_with_upwards_trend: Bullish Sentiment Moving Average")
    ma_window = st.slider("Select MA Window (weeks):", 1, 52, 52, key="tab2_ma")
    df_ma = filtered_df.copy()
    df_ma["Bullish_MA"] = df_ma["Bullish"].rolling(window=ma_window, min_periods=1).mean()
    base_ma = alt.Chart(df_ma).encode(x='Date:T')
    chart3 = alt.layer(
        base_ma.mark_line(color='black').encode(y=alt.Y('SP500_Close:Q', title='S&P 500 Price')),
        base_ma.mark_line(color='green').encode(y=alt.Y('Bullish_MA:Q', title='Bullish Sentiment MA'))
    ).resolve_scale(y='independent').properties(height=300)
    st.altair_chart(chart3, use_container_width=True)





# ---------------------------- TAB 4 ----------------------------
with tab4:
    import plotly.graph_objects as go

    st.markdown("## 🧐 CNN-Style Fear & Greed Replication")

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

    # F&G Index
    fig_fng.add_trace(go.Scatter(
        x=fng_df.index, y=fng_df["FNG_Index"],
        name="F&G Index", mode="lines", yaxis="y1"
    ))

    # 100-day MA
    fig_fng.add_trace(go.Scatter(
        x=fng_df.index, y=fng_df["FNG_Smooth"],
        name="100-day MA", mode="lines", line=dict(color="red"), yaxis="y1"
    ))

    # S&P 500 or SPY price
    spy_price = data["SPY"].reindex(fng_df.index)
    fig_fng.add_trace(go.Scatter(
        x=spy_price.index, y=spy_price,
        name="SPY Price", mode="lines", line=dict(color="gray", dash="dot"), yaxis="y2"
    ))

    fig_fng.update_layout(
        yaxis=dict(title="F&G Index (0–100)", range=[0, 100]),
        yaxis2=dict(
            title="SPY Price", overlaying="y", side="right", showgrid=False
        ),
        xaxis=dict(title="Date"),
        shapes=[
            dict(type="rect", xref="x", yref="y", x0=fng_df.index[0], x1=fng_df.index[-1], y0=0, y1=25,
                 fillcolor="#ffcccc", opacity=0.2, line_width=0),
            dict(type="rect", xref="x", yref="y", x0=fng_df.index[0], x1=fng_df.index[-1], y0=75, y1=100,
                 fillcolor="#d9f2d9", opacity=0.2, line_width=0),
        ],
        height=600,
        margin=dict(l=40, r=40, t=30, b=30)
    )

    st.plotly_chart(fig_fng, use_container_width=True)

    # 📊 Distribution + stats below
    st.markdown("### 📊 Distribution of the F&G Index (2007–Today)")
    st.markdown("This histogram shows the distribution of the CNN-style Fear & Greed Index since 2007. Vertical lines mark key sentiment thresholds.")

    col_left, col_right = st.columns([3, 1])

    with col_left:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=fng_df["FNG_Index"],
            nbinsx=50,
            marker_color='rgba(100, 150, 250, 0.7)',
            opacity=0.75
        ))

        for val, label in zip([25, 50, 75], ["Extreme Fear", "Neutral", "Greed"]):
            fig_hist.add_shape(
                type="line", x0=val, x1=val, y0=0, y1=1, xref="x", yref="paper",
                line=dict(color="black", dash="dot")
            )
            fig_hist.add_annotation(
                x=val, y=1, yref="paper", text=label,
                showarrow=False, font=dict(size=12), yanchor="bottom"
            )

        fig_hist.update_layout(
            xaxis_title="F&G Index Value",
            yaxis_title="Frequency",
            height=500,
            bargap=0.05
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        stats = {
            "Mean": fng_df["FNG_Index"].mean(),
            "Std. Dev.": fng_df["FNG_Index"].std(),
            "Min": fng_df["FNG_Index"].min(),
            "Max": fng_df["FNG_Index"].max(),
            "Skewness": fng_df["FNG_Index"].skew(),
            "Kurtosis": fng_df["FNG_Index"].kurtosis()
        }
        st.markdown("#### 📋 Summary Stats")
        st.dataframe(pd.DataFrame(stats, index=["Value"]).T, use_container_width=True)
# ---------------- tab 5 ----------------
with tab5:
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


# ---------------------------- TAB 6 ----------------------------------
with tab6:
    st.markdown("""
### 🧠 Distribution of SPY Returns with Historical & F&G-Adjusted VaR:  
This histogram visualizes the **distribution of daily SPY returns** over the past 5 years.

🔹 The bars represent how frequently specific return levels occurred.

🔹 The vertical lines overlay different **Value-at-Risk (VaR)** and **Conditional VaR (CVaR)** methods:
- **Historical VaR / CVaR**: Based purely on percentiles from past returns  
- **Parametric (Normal)**: Assumes returns follow a normal distribution  
- **Monte Carlo**: Simulates thousands of random returns based on observed mean and volatility  
- **F&G-Adjusted**: Dynamically adjusts the risk threshold depending on sentiment:
  - More fear → higher risk buffer
  - More greed → tighter buffer

This shows how **risk thresholds shift depending on method**, and how **sentiment-adjusted VaR can better reflect market emotions**.
""")

    investment = 1_000_000
    confidence_level = 0.95
    alpha = 1 - confidence_level
    lookback_days = 252 * 5

    spy_prices = data["SPY"].dropna()
    spy_returns = spy_prices.pct_change().dropna()
    spy_returns = spy_returns[-lookback_days:]

    # === Static VaR & CVaR
    sorted_returns = np.sort(spy_returns.values)
    var_hist = np.percentile(sorted_returns, alpha * 100)
    cvar_hist = sorted_returns[sorted_returns <= var_hist].mean()

    mu = spy_returns.mean()
    sigma = spy_returns.std()
    from scipy.stats import norm
    var_param = norm.ppf(alpha, mu, sigma)
    cvar_param = mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha

    np.random.seed(42)
    sim_returns = np.random.normal(mu, sigma, 100000)
    var_mc = np.percentile(sim_returns, alpha * 100)
    cvar_mc = sim_returns[sim_returns <= var_mc].mean()

    # === Rolling Window Length
    st.markdown("### 📏 Select Rolling Window Length")
    window = st.slider("Rolling Window (days)", min_value=100, max_value=500, value=252, step=10)

    # === F&G Adjusted Alpha (only sentiment)
    full_returns = data["SPY"].pct_change().dropna()
    fng_series = fng_df["FNG_Index"].reindex(full_returns.index).dropna()
    full_returns = full_returns.loc[fng_series.index]

    fng_alpha = 0.01 + ((100 - fng_series) / 100) * 0.09
    fng_alpha = fng_alpha.clip(0.01, 0.2)

    # === Compute Adjusted VaR & CVaR
    adjusted_var = pd.Series(index=full_returns.index, dtype=float)
    adjusted_cvar = pd.Series(index=full_returns.index, dtype=float)
    for date in full_returns.index[window:]:
        past = full_returns.loc[:date].iloc[-window:]
        alpha_t = fng_alpha.loc[date]
        v = np.percentile(past, alpha_t * 100)
        cv = past[past <= v].mean()
        adjusted_var.loc[date] = v
        adjusted_cvar.loc[date] = cv

    # === Rolling Historical VaR/CVaR
    rolling_var = full_returns.rolling(window).quantile(alpha)
    rolling_cvar = full_returns.rolling(window).apply(lambda x: x[x <= x.quantile(alpha)].mean(), raw=False)

    # === Histogram
    latest_adj_var = adjusted_var.dropna().iloc[-1]
    latest_adj_cvar = adjusted_cvar.dropna().iloc[-1]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=spy_returns * 100, nbinsx=100, name="SPY Returns", marker_color="#cce5ff", opacity=0.75))
    fig.add_trace(go.Scatter(x=[var_hist * 100]*2, y=[0, 100], name="VaR (Historical)", line=dict(color="#66b3ff")))
    fig.add_trace(go.Scatter(x=[var_param * 100]*2, y=[0, 100], name="VaR (Parametric)", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=[var_mc * 100]*2, y=[0, 100], name="VaR (Monte Carlo)", line=dict(color="orange", dash="dot")))
    fig.add_trace(go.Scatter(x=[cvar_hist * 100]*2, y=[0, 100], name="CVaR (Historical)", line=dict(color="#004080", dash="dot")))
    fig.add_trace(go.Scatter(x=[cvar_param * 100]*2, y=[0, 100], name="CVaR (Parametric)", line=dict(color="darkgreen", dash="dot")))
    fig.add_trace(go.Scatter(x=[cvar_mc * 100]*2, y=[0, 100], name="CVaR (Monte Carlo)", line=dict(color="darkorange", dash="dot")))
    fig.add_trace(go.Scatter(x=[latest_adj_var * 100]*2, y=[0, 100], name="F&G Adjusted VaR", line=dict(color="#ff6666", dash="dot")))
    fig.add_trace(go.Scatter(x=[latest_adj_cvar * 100]*2, y=[0, 100], name="F&G Adjusted CVaR", line=dict(color="#800000", dash="dot")))
    fig.update_layout(title="Distribution of SPY Returns with Historical & F&G Adjusted VaR", height=600)

    # === Time Series Comparison with S&P 500 Price
    indexed_price = (spy_prices / spy_prices.loc[adjusted_var.dropna().index[0]]) * 100
    indexed_price = indexed_price.reindex(adjusted_var.index)

    fig_combined = go.Figure()
    fig_combined.add_trace(go.Scatter(x=rolling_var.index, y=rolling_var * 100, name="Historical VaR", line=dict(color="#66b3ff")))
    fig_combined.add_trace(go.Scatter(x=rolling_cvar.index, y=rolling_cvar * 100, name="Historical CVaR", line=dict(color="#004080")))
    fig_combined.add_trace(go.Scatter(x=adjusted_var.index, y=adjusted_var * 100, name="F&G Adjusted VaR", line=dict(color="#ff6666", dash="dot")))
    fig_combined.add_trace(go.Scatter(x=adjusted_cvar.index, y=adjusted_cvar * 100, name="F&G Adjusted CVaR", line=dict(color="#800000", dash="dot")))
    fig_combined.add_trace(go.Scatter(x=indexed_price.index, y=indexed_price, name="S&P 500 Indexed", line=dict(color="black", width=1.5), yaxis="y2"))
    fig_combined.update_layout(
        title="📉 Rolling VaR & CVaR vs S&P 500 Price (Indexed)",
        xaxis=dict(title="Date"),
        yaxis=dict(title="VaR / CVaR (%)"),
        yaxis2=dict(title="S&P 500 (Indexed)", overlaying="y", side="right", showgrid=False),
        height=600,
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    # === Display Layout
    col1, col2 = st.columns([4, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("### 📊 VaR Table")
        summary_df = pd.DataFrame({
            "VaR (%)": [var_hist * 100, var_param * 100, var_mc * 100],
            "CVaR (%)": [cvar_hist * 100, cvar_param * 100, cvar_mc * 100],
            "VaR ($)": [-var_hist * investment, -var_param * investment, -var_mc * investment],
            "CVaR ($)": [-cvar_hist * investment, -cvar_param * investment, -cvar_mc * investment]
        }, index=["Historical", "Parametric", "Monte Carlo"])
        st.dataframe(summary_df.round(2), use_container_width=True, height=350)
    st.markdown("""
### 🧮 F&G Adjusted VaR Formula:

This time-series plot compares **rolling VaR/CVaR** estimates against the **indexed S&P 500 price**.

There are two types of risk estimates:
- **Traditional (Historical)**: Fixed confidence level (e.g. 95%) applied over a rolling window
- **F&G-Adjusted**: Varies the alpha threshold **based on market sentiment** via the Fear & Greed Index.

By overlaying the S&P 500 price:
- You can observe **how risk estimates evolve alongside market trends**
- Notice that F&G-adjusted risk **spikes during fearful periods** (when prices are falling)
- This allows portfolios to **adapt exposure dynamically to changing emotional regimes**

💡 The dual-axis format helps highlight **timing relationships between market downturns and risk model responsiveness**.
""")

    st.markdown(r"""
    $$ \alpha(t) = 0.01 + \left( \frac{100 - \text{F\&G}(t)}{100} \right) \cdot 0.09 $$

    - Higher fear → higher α(t) → higher VaR
    - Greedy market → α closer to 1% → lower VaR
    """)

    col3, col4 = st.columns([4, 1])
    with col3:
        st.plotly_chart(fig_combined, use_container_width=True)
    with col4:
        st.markdown("### ❗ Breach Frequency")
        breach_df = pd.DataFrame({
            "Historical VaR Breaches": (full_returns.loc[rolling_var.index] < rolling_var).mean() * 100,
            "F&G Adj. VaR Breaches": (full_returns.loc[adjusted_var.index] < adjusted_var).mean() * 100,
        }, index=["% of Days"]).T
        st.dataframe(breach_df.round(2), use_container_width=True)

# ---------------------------- TAB 7 ----------------------------------
with tab7:
    st.markdown("## ⚖️ F&G-Adjusted Stop-Loss with Bullish Sentiment Re-entry (60/40 Portfolio)")
    st.markdown("""
### 🧠 Strategy Overview

This strategy dynamically reduces portfolio exposure during high-risk periods using:
- **Value at Risk (VaR)**: 5% rolling quantile of recent 100-day returns
- **Fear & Greed (F&G) Index**: Determines how aggressively to scale back exposure:
  - Lower sentiment → higher stop-loss multiplier
  - Higher sentiment → tighter thresholds

When a return breach occurs (i.e., below the F&G-adjusted VaR), exposure drops to **30%**.
It only returns to normal levels when:
- Bullish sentiment rises above a user-defined threshold (default: 30%)
- At least 3 quiet days have passed

### 🔍 Interpretation of the Chart
- The plot compares **indexed performance** of the raw 60/40 portfolio and the **strategy with stop-loss and bullish sentiment re-entry**
- The strategy aims to **limit drawdowns** during downturns and re-enter at favorable sentiment conditions

### 📟 Trigger Table
- The second table breaks down how many stop-loss events were triggered per year and regime
  (Extreme Fear, Fear, Greed, Extreme Greed)

### 📋 Summary Table
- This compares performance metrics:
  - Return
  - Volatility
  - Conditional VaR (CVaR)
  - Downside Deviation
  - Max Drawdown

The goal: **enhance downside protection** while participating in upside trends using a **sentiment-aware risk overlay**.
""")


    # Add adjustable start date selector
    available_dates = data.index.intersection(fng_df.index)
    default_start = available_dates[0].date()
    selected_start = st.date_input("Select portfolio start date:", value=default_start, min_value=default_start, max_value=available_dates[-1].date())
    selected_start = pd.to_datetime(selected_start)

    exposure_floor = 0.3
    exposure_ceiling = 1.0
    bullish_threshold = st.slider("Minimum Bullish Sentiment to Re-enter Market (%)", 0, 100, 30)

    spy = data["SPY"].pct_change()
    tlt = data["TLT"].pct_change()
    fng_series = fng_df["FNG_Index"]

    common_idx = spy.dropna().index.intersection(tlt.dropna().index).intersection(fng_series.dropna().index)
    common_idx = common_idx[common_idx >= selected_start]  # Apply selected start date

    spy = spy.loc[common_idx]
    tlt = tlt.loc[common_idx]
    fng_series = fng_series.loc[common_idx]
    port_returns = (0.6 * spy + 0.4 * tlt).dropna()

    var_series = port_returns.rolling(window=100).apply(lambda x: np.percentile(x, 5)).dropna()
    common_idx = port_returns.index.intersection(var_series.index)
    port_returns = port_returns.loc[common_idx]
    var_series = var_series.loc[common_idx]
    fng_series = fng_series.loc[common_idx]

    def stop_loss_multiplier_from_fng(fng):
        if fng < 25: return 1.5
        elif fng < 50: return 1.2
        elif fng < 75: return 1.0
        else: return 0.8

    sl_multiplier = fng_series.apply(stop_loss_multiplier_from_fng)
    stop_loss_threshold = var_series * sl_multiplier
    triggered = port_returns < stop_loss_threshold

    bullish_series = clean_df.set_index("Date")["Bullish"].resample("D").ffill()
    bullish_series = bullish_series.reindex(port_returns.index, method="ffill")

    var_scaled = (var_series - var_series.min()) / (var_series.max() - var_series.min())
    dynamic_exposure = exposure_ceiling - var_scaled * (exposure_ceiling - exposure_floor)
    dynamic_exposure = dynamic_exposure.clip(exposure_floor, exposure_ceiling)

    exposure_series = pd.Series(index=port_returns.index, dtype=float)
    for i, date in enumerate(port_returns.index):
        if i == 0:
            exposure_series.iloc[i] = dynamic_exposure.iloc[i]
        else:
            if triggered.iloc[i]:
                exposure_series.iloc[i] = exposure_floor
            else:
                if bullish_series.iloc[i] >= bullish_threshold:
                    exposure_series.iloc[i] = dynamic_exposure.iloc[i]
                else:
                    exposure_series.iloc[i] = exposure_floor

    adjusted_returns = port_returns * exposure_series.shift(1).fillna(exposure_ceiling)
    cumulative_return = (1 + adjusted_returns).cumprod()
    cumulative_no_sl = (1 + port_returns).cumprod()

    st.markdown("### 📈 60/40 Portfolio: Raw vs Stop-Loss + Bullish Filter")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative_no_sl.index, y=cumulative_no_sl / cumulative_no_sl.iloc[0], name="60/40 No Stop-Loss"))
    fig.add_trace(go.Scatter(x=cumulative_return.index, y=cumulative_return / cumulative_return.iloc[0], name="60/40 with SL + Bullish Re-entry"))
    fig.update_layout(title="Indexed Portfolio Value", yaxis_title="Value", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📟 Stop-Loss Trigger Stats by Sentiment Regime")

    def classify_regime(fng):
        if fng < 25: return "Extreme Fear"
        elif fng < 50: return "Fear"
        elif fng < 75: return "Greed"
        else: return "Extreme Greed"

    regime_series = fng_series.apply(classify_regime)
    trigger_df = pd.DataFrame({
        "Year": triggered.index.year,
        "Triggered": triggered,
        "Regime": regime_series
    })

    summary = (
        trigger_df[trigger_df["Triggered"]]
        .groupby(["Year", "Regime"])
        .size()
        .unstack(fill_value=0)
    )

    st.dataframe(summary)

    st.markdown("### 📋 Strategy Summary Table")

    def max_drawdown(cum):
        roll_max = cum.cummax()
        return (cum / roll_max - 1).min()

    strat_r = adjusted_returns
    naive_r = port_returns

    stats_summary = pd.DataFrame({
        "Return (%)": [
            (cumulative_no_sl.iloc[-1] / cumulative_no_sl.iloc[0] - 1) * 100,
            (cumulative_return.iloc[-1] / cumulative_return.iloc[0] - 1) * 100
        ],
        "Volatility (%)": [
            naive_r.std() * np.sqrt(252) * 100,
            strat_r.std() * np.sqrt(252) * 100
        ],
        "CVaR (95%) (%)": [
            naive_r[naive_r < np.percentile(naive_r, 5)].mean() * 100,
            strat_r[strat_r < np.percentile(strat_r, 5)].mean() * 100
        ],
        "Downside Dev. (%)": [
            np.sqrt(np.mean(np.minimum(0, naive_r) ** 2)) * np.sqrt(252) * 100,
            np.sqrt(np.mean(np.minimum(0, strat_r) ** 2)) * np.sqrt(252) * 100
        ],
        "Max Drawdown (%)": [
            max_drawdown(cumulative_no_sl) * 100,
            max_drawdown(cumulative_return) * 100
        ]
    }, index=["60/40 Only", "With SL + Bullish Re-entry"])

    st.dataframe(stats_summary.round(2), use_container_width=True)


# ---------------------------- TAB 8 ----------------------------------

with tab8:
    st.markdown("## 🧨 F&G + Bullish-Adjusted Stop-Loss Performance During Crises (60/40 SPY/TLT)")

    crisis_periods = {
        "2008 Crash": ("2008-09-01", "2009-04-01"),
        "COVID Crash": ("2020-02-01", "2020-07-01"),
        "2022 Bear Market": ("2022-01-01", "2023-01-01")
    }

    spy = data["SPY"].pct_change()
    tlt = data["TLT"].pct_change()
    vix = data["^VIX"].pct_change().fillna(0)
    fng_series = fng_df["FNG_Index"]
    bullish_series = load_clean_data().set_index("Date")["Bullish"].reindex(spy.index).fillna(method="ffill")

    common_idx = spy.dropna().index.intersection(tlt.dropna().index).intersection(fng_series.dropna().index)
    spy = spy.loc[common_idx]
    tlt = tlt.loc[common_idx]
    vix = vix.loc[common_idx]
    fng_series = fng_series.loc[common_idx]
    bullish_series = bullish_series.loc[common_idx]

    port_returns = (0.6 * spy + 0.4 * tlt)
    var_series = port_returns.rolling(100).apply(lambda x: np.percentile(x, 5)).dropna()
    var_series = var_series.reindex(port_returns.index, method="ffill")
    fng_series = fng_series.reindex(port_returns.index, method="ffill")

    def stop_loss_multiplier(fng):
        if fng < 25: return 1.5
        elif fng < 50: return 1.2
        elif fng < 75: return 1.0
        else: return 0.8

    sl_multiplier = fng_series.apply(stop_loss_multiplier)
    threshold = var_series * sl_multiplier
    triggered = port_returns < threshold

    min_bullish_to_reenter = 40
    exposure = pd.Series(index=port_returns.index, dtype=float)
    exposure.iloc[0] = 1.0
    quiet_days = 0
    for i in range(1, len(port_returns)):
        if triggered.iloc[i]:
            exposure.iloc[i] = 0.3
            quiet_days = 0
        else:
            quiet_days += 1
            if quiet_days >= 3 and bullish_series.iloc[i] >= min_bullish_to_reenter:
                exposure.iloc[i] = 1.0
            else:
                exposure.iloc[i] = 0.3

    strategy_returns = port_returns * exposure.shift(1).fillna(1.0)
    cum_strategy = (1 + strategy_returns).cumprod()
    cum_naive = (1 + port_returns).cumprod()

    vix_exposure = vix.apply(lambda x: 0.3 if x > 0.02 else 1.0)
    vix_strategy_returns = port_returns * vix_exposure.shift(1).fillna(1.0)
    cum_vix_strategy = (1 + vix_strategy_returns).cumprod()

    st.markdown("### 📈 Last 6 Months: Indexed Portfolio Comparison")

    aligned_index = cum_strategy.index.intersection(cum_naive.index).intersection(cum_vix_strategy.index)
    last_6mo_start = aligned_index[-126]
    sub_index = aligned_index[aligned_index >= last_6mo_start]
    start_idx = sub_index[0]

    fig_last6mo = go.Figure()
    fig_last6mo.add_trace(go.Scatter(
        x=sub_index,
        y=cum_naive.loc[sub_index] / cum_naive.loc[start_idx],
        name="60/40 Portfolio",
        line=dict(color="navy")
    ))
    fig_last6mo.add_trace(go.Scatter(
        x=sub_index,
        y=cum_strategy.loc[sub_index] / cum_strategy.loc[start_idx],
        name="With F&G + Bullish Stop-Loss",
        line=dict(color="skyblue")
    ))
    fig_last6mo.add_trace(go.Scatter(
        x=sub_index,
        y=cum_vix_strategy.loc[sub_index] / cum_vix_strategy.loc[start_idx],
        name="With VIX Strategy",
        line=dict(dash="dash", color="orange")
    ))
    fig_last6mo.update_layout(
        title="Last 6 Months: Indexed Portfolio Comparison",
        yaxis_title="Indexed Value",
        height=400,
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig_last6mo, use_container_width=True)

    naive_r = port_returns.loc[start_idx:]
    strat_r = strategy_returns.loc[start_idx:]
    vix_r = vix_strategy_returns.loc[start_idx:]

    def max_drawdown(cum):
        roll_max = cum.cummax()
        drawdown = cum / roll_max - 1.0
        return drawdown.min()

    stats_6mo = pd.DataFrame({
        "Return (%)": [
            (cum_naive.loc[sub_index[-1]] / cum_naive.loc[start_idx] - 1) * 100,
            (cum_strategy.loc[sub_index[-1]] / cum_strategy.loc[start_idx] - 1) * 100,
            (cum_vix_strategy.loc[sub_index[-1]] / cum_vix_strategy.loc[start_idx] - 1) * 100
        ],
        "Volatility (%)": [
            naive_r.std() * np.sqrt(252) * 100,
            strat_r.std() * np.sqrt(252) * 100,
            vix_r.std() * np.sqrt(252) * 100
        ],
        "CVaR (95%) (%)": [
            naive_r[naive_r < np.percentile(naive_r, 5)].mean() * 100,
            strat_r[strat_r < np.percentile(strat_r, 5)].mean() * 100,
            vix_r[vix_r < np.percentile(vix_r, 5)].mean() * 100
        ],
        "Downside Dev. (%)": [
            np.sqrt(np.mean(np.minimum(0, naive_r)**2)) * np.sqrt(252) * 100,
            np.sqrt(np.mean(np.minimum(0, strat_r)**2)) * np.sqrt(252) * 100,
            np.sqrt(np.mean(np.minimum(0, vix_r)**2)) * np.sqrt(252) * 100
        ],
        "Max Drawdown (%)": [
            max_drawdown(cum_naive.loc[sub_index]) * 100,
            max_drawdown(cum_strategy.loc[sub_index]) * 100,
            max_drawdown(cum_vix_strategy.loc[sub_index]) * 100
        ]
    }, index=["60/40 Only", "With F&G Stop-Loss", "With VIX Strategy"])

    st.dataframe(stats_6mo.round(2))

    for label, (start, end) in crisis_periods.items():
        st.markdown(f"### 📉 {label}")
        try:
            sub_index = cum_strategy.loc[start:end].index
            start_idx = sub_index[0]
            end_idx = sub_index[-1]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sub_index, y=cum_naive.loc[sub_index] / cum_naive.loc[start_idx],
                name="60/40 Portfolio"
            ))
            fig.add_trace(go.Scatter(
                x=sub_index, y=cum_strategy.loc[sub_index] / cum_strategy.loc[start_idx],
                name="With F&G + Bullish Stop-Loss"
            ))
            fig.add_trace(go.Scatter(
                x=sub_index, y=cum_vix_strategy.loc[sub_index] / cum_vix_strategy.loc[start_idx],
                name="With VIX Strategy",
                line=dict(dash="dash", color="orange")
            ))
            fig.update_layout(title=f"Performance Comparison During {label}", yaxis_title="Indexed Value", height=400)
            st.plotly_chart(fig, use_container_width=True)

            naive_r = port_returns.loc[start_idx:end_idx]
            strat_r = strategy_returns.loc[start_idx:end_idx]
            vix_r = vix_strategy_returns.loc[start_idx:end_idx]

            stats = pd.DataFrame({
                "Return (%)": [
                    (cum_naive.loc[end_idx] / cum_naive.loc[start_idx] - 1) * 100,
                    (cum_strategy.loc[end_idx] / cum_strategy.loc[start_idx] - 1) * 100,
                    (cum_vix_strategy.loc[end_idx] / cum_vix_strategy.loc[start_idx] - 1) * 100
                ],
                "Volatility (%)": [
                    naive_r.std() * np.sqrt(252) * 100,
                    strat_r.std() * np.sqrt(252) * 100,
                    vix_r.std() * np.sqrt(252) * 100
                ],
                "CVaR (95%) (%)": [
                    naive_r[naive_r < np.percentile(naive_r, 5)].mean() * 100,
                    strat_r[strat_r < np.percentile(strat_r, 5)].mean() * 100,
                    vix_r[vix_r < np.percentile(vix_r, 5)].mean() * 100
                ],
                "Downside Dev. (%)": [
                    np.sqrt(np.mean(np.minimum(0, naive_r)**2)) * np.sqrt(252) * 100,
                    np.sqrt(np.mean(np.minimum(0, strat_r)**2)) * np.sqrt(252) * 100,
                    np.sqrt(np.mean(np.minimum(0, vix_r)**2)) * np.sqrt(252) * 100
                ],
                "Max Drawdown (%)": [
                    max_drawdown(cum_naive.loc[start_idx:end_idx]) * 100,
                    max_drawdown(cum_strategy.loc[start_idx:end_idx]) * 100,
                    max_drawdown(cum_vix_strategy.loc[start_idx:end_idx]) * 100
                ]
            }, index=["60/40 Only", "With F&G Stop-Loss", "With VIX Strategy"])

            st.dataframe(stats.round(2))
        except Exception as e:
            st.warning(f"⚠️ Skipping {label} due to data alignment issue: {e}")



# ---------------------------- TAB 9 ----------------------------------
with tab9:
    st.markdown("## 📊 Full-Period Summary Metrics: Dynamic Weights + Dynamic Exposure")

    # 📘 Explanation block
    st.markdown("""
    ### ⚙️ Strategy Description

    This version combines two layers of risk management:
    - **Dynamic Weight Allocation**: SPY and TLT weights shift based on the Fear & Greed Index:
        - **F&G < 25**: 30% SPY / 70% TLT (defensive)
        - **25 ≤ F&G < 50**: 50% / 50%
        - **50 ≤ F&G < 75**: 70% / 30%
        - **F&G ≥ 75**: 85% / 15% (aggressive)
    - **Dynamic Exposure Control**: Overall exposure to the portfolio is scaled down when risk is elevated:
        - Exposure is reduced when returns breach an F&G-adjusted VaR
        - Re-entry allowed after 3 quiet days if **bullish sentiment ≥ 40%**
    """)

    # === Portfolio start date selector
    min_date = pd.to_datetime("2007-01-01")
    max_date = pd.to_datetime("today")
    start_date = st.date_input("📅 Portfolio start date:", value=min_date, min_value=min_date, max_value=max_date)
    start_date = pd.to_datetime(start_date)

    # === Data
    spy = data["SPY"].pct_change()
    tlt = data["TLT"].pct_change()
    fng_series = fng_df["FNG_Index"]
    bullish_series = load_clean_data().set_index("Date")["Bullish"].reindex(spy.index).fillna(method="ffill")

    common_idx = spy.dropna().index.intersection(tlt.dropna().index).intersection(fng_series.dropna().index)
    common_idx = common_idx[common_idx >= start_date]

    spy = spy.loc[common_idx]
    tlt = tlt.loc[common_idx]
    fng_series = fng_series.loc[common_idx]
    bullish_series = bullish_series.loc[common_idx]

    # === Dynamic Weights
    def get_weights(fng):
        if fng < 25: return 0.3, 0.7
        elif fng < 50: return 0.5, 0.5
        elif fng < 75: return 0.7, 0.3
        else: return 0.85, 0.15

    weights = fng_series.apply(lambda x: pd.Series(get_weights(x), index=["w_spy", "w_tlt"]))
    w_spy = weights["w_spy"]
    w_tlt = weights["w_tlt"]

    # === Portfolio returns
    port_returns = (spy * w_spy + tlt * w_tlt).dropna()

    # === VaR-based stop-loss
    var_series = port_returns.rolling(100).apply(lambda x: np.percentile(x, 5)).dropna()
    var_series = var_series.reindex(port_returns.index, method="ffill")
    fng_series = fng_series.reindex(port_returns.index, method="ffill")

    def stop_loss_multiplier(fng):
        if fng < 25: return 1.5
        elif fng < 50: return 1.2
        elif fng < 75: return 1.0
        else: return 0.8

    sl_multiplier = fng_series.apply(stop_loss_multiplier)
    threshold = var_series * sl_multiplier
    triggered = port_returns < threshold

    # === Exposure control
    min_bullish = 40
    exposure = pd.Series(index=port_returns.index, dtype=float)
    exposure.iloc[0] = 1.0
    quiet_days = 0
    scaled_exposure = (threshold - threshold.min()) / (threshold.max() - threshold.min())
    scaled_exposure = 1 - scaled_exposure  # invert: high VaR → low exposure

    for i in range(1, len(port_returns)):
        if triggered.iloc[i]:
            exposure.iloc[i] = scaled_exposure.iloc[i] * 0.7 + 0.3
            quiet_days = 0
        else:
            quiet_days += 1
            if quiet_days >= 3 and bullish_series.iloc[i] >= min_bullish:
                exposure.iloc[i] = 1.0
            else:
                exposure.iloc[i] = exposure.iloc[i - 1]

    # === Final strategy returns
    strategy_returns = port_returns * exposure.shift(1).fillna(1.0)
    cum_strategy = (1 + strategy_returns).cumprod()

    # === Benchmark
    static_returns = (0.6 * spy + 0.4 * tlt).reindex(cum_strategy.index)
    cum_static = (1 + static_returns).cumprod()

    # === Risk Metrics
    def max_drawdown(cum): return (cum / cum.cummax() - 1).min()
    sharpe_ratio = lambda r: (r.mean() / r.std()) * np.sqrt(252)

    strat_r = strategy_returns
    naive_r = static_returns

    stats_all = pd.DataFrame({
        "Return (%)": [
            (cum_static.iloc[-1] / cum_static.iloc[0] - 1) * 100,
            (cum_strategy.iloc[-1] / cum_strategy.iloc[0] - 1) * 100
        ],
        "Volatility (%)": [
            naive_r.std() * np.sqrt(252) * 100,
            strat_r.std() * np.sqrt(252) * 100
        ],
        "CVaR (95%) (%)": [
            naive_r[naive_r < np.percentile(naive_r, 5)].mean() * 100,
            strat_r[strat_r < np.percentile(strat_r, 5)].mean() * 100
        ],
        "Downside Dev. (%)": [
            np.sqrt(np.mean(np.minimum(0, naive_r) ** 2)) * np.sqrt(252) * 100,
            np.sqrt(np.mean(np.minimum(0, strat_r) ** 2)) * np.sqrt(252) * 100
        ],
        "Max Drawdown (%)": [
            max_drawdown(cum_static) * 100,
            max_drawdown(cum_strategy) * 100
        ],
        "Sharpe Ratio": [
            sharpe_ratio(naive_r),
            sharpe_ratio(strat_r)
        ]
    }, index=["60/40 Only", "F&G Dynamic Weights + SL"])

    # === Graphs ===
    st.markdown(f"### 📊 Dynamic Portfolio Behavior Since {start_date.date()}")

    col1, col2 = st.columns(2)
    with col1:
        fig_alloc = go.Figure()
        fig_alloc.add_trace(go.Scatter(x=w_spy.index, y=w_spy, mode="lines", stackgroup="one", name="SPY Weight", line=dict(color="skyblue")))
        fig_alloc.add_trace(go.Scatter(x=w_tlt.index, y=w_tlt, mode="lines", stackgroup="one", name="TLT Weight", line=dict(color="lightgray")))
        fig_alloc.update_layout(title="🧠 SPY/TLT Weight Allocation Over Time", height=400, yaxis_title="Weight", xaxis_title="Date", showlegend=True)
        st.plotly_chart(fig_alloc, use_container_width=True)

    with col2:
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=cum_static.index, y=cum_static / cum_static.iloc[0], name="60/40 Portfolio", line=dict(color="navy")))
        fig_perf.add_trace(go.Scatter(x=cum_strategy.index, y=cum_strategy / cum_strategy.iloc[0], name="F&G Dyn. Strategy", line=dict(color="dodgerblue")))
        fig_perf.update_layout(title="📈 Full Period Indexed Performance", height=400, yaxis_title="Indexed Value")
        st.plotly_chart(fig_perf, use_container_width=True)

    # === Radar chart for risk
    import plotly.express as px
    radar_df = stats_all[["Volatility (%)", "CVaR (95%) (%)", "Downside Dev. (%)", "Max Drawdown (%)"]]
    fig_radar = px.line_polar(
        radar_df.T,
        r=radar_df.T.values,
        theta=radar_df.T.index,
        line_close=True,
        labels={"variable": "Strategy"},
        title="🕸️ Risk Profile Comparison (Lower = Better)"
    )
    fig_radar.update_traces(fill="toself")
    st.plotly_chart(fig_radar, use_container_width=True)

    # === Color-coded summary table
    def highlight_risks(val, metric):
        if metric in ["Volatility (%)", "CVaR (95%) (%)", "Downside Dev. (%)", "Max Drawdown (%)"]:
            return "background-color: #c6efce" if val < 10 else "background-color: #ffc7ce"
        elif metric == "Sharpe Ratio":
            return "background-color: #c6efce" if val > 1 else "background-color: #ffc7ce"
        return ""

    st.markdown("### 📋 Summary Risk Metrics Table")
    styled = stats_all.style.apply(lambda x: [highlight_risks(v, x.index[i]) for i, v in enumerate(x)], axis=1)
    st.dataframe(styled, use_container_width=True)
