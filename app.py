import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from datetime import datetime

# Load and cache data
@st.cache_data
def load_data():
    file_path = "sentiment_data.xlsx"
    df = pd.read_excel(file_path, skiprows=4)
    df = df.rename(columns={
        df.columns[0]: "Date",
        df.columns[1]: "Bullish",
        df.columns[2]: "Neutral",
        df.columns[3]: "Bearish",
        df.columns[12]: "SP500_Close"
    })
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.astype({"Bullish": float, "Neutral": float, "Bearish": float, "SP500_Close": float})
    df["SP500_Return"] = df["SP500_Close"].pct_change() * 100
    return df.dropna()

# App layout
st.set_page_config(layout="wide")
st.title("📊 AAII Sentiment & S&P 500 Dashboard")

# Load data
df = load_data()

tab1, tab2, tab3 = st.tabs(["📄 Raw Excel Viewer", "📈 Interactive Dashboard", "📘 Factor Model"])

# Tab 1: Raw Excel Viewer
with tab1:
    st.subheader("🗂️ Raw AAII Sentiment Excel File")
    st.dataframe(df.reset_index(), height=400)

# Tab 2: Interactive Dashboard
with tab2:
    st.subheader("📅 Select Time Range for Analysis")
    min_date, max_date = df.index.min(), df.index.max()
    start_date, end_date = st.slider("Select Date Range:", min_value=min_date, max_value=max_date,
                                     value=(min_date, max_date), format="YYYY-MM-DD")
    filtered_df = df.loc[start_date:end_date]

    st.subheader("📉 S&P 500 Weekly Close (Log Scale)")
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(filtered_df.index, filtered_df["SP500_Close"], color="black")
    ax1.set_yscale("log")
    ax1.set_title("S&P 500 Weekly Close")
    ax1.set_ylabel("Price")
    st.pyplot(fig1)

    st.subheader("🧠 AAII Sentiment Over Time")
    sentiment_options = ["Bullish", "Neutral", "Bearish"]
    show_sentiments = st.multiselect("Select Sentiment Components", sentiment_options, default=sentiment_options,
                                     format_func=lambda x: {"Bullish": "🐂 Bullish", "Neutral": "😐 Neutral", "Bearish": "🐻 Bearish"}[x])
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    for sentiment in show_sentiments:
        ax2.plot(filtered_df.index, filtered_df[sentiment], label=sentiment)
    ax2.set_title("Investor Sentiment")
    ax2.set_ylabel("Sentiment (%)")
    ax2.legend()
    st.pyplot(fig2)

# Tab 3: Factor Model
with tab3:
    st.subheader("📘 Factor Model: Estimating Factor Loadings via Regression")
    st.markdown(r"""
    This model estimates how weekly S&P 500 returns respond to observable sentiment factors. We apply a classical regression-based calibration as described in the factor model literature.

    **Model**: \( R_t = \beta_0 + \beta_1 \cdot \text{Bullish}_{t-1} + \beta_2 \cdot \text{Bearish}_{t-1} + \varepsilon_t \)

    In matrix form: \( X = FB + E \Rightarrow \hat{B} = (F^\top F)^{-1}F^\top X \)
    """)

    # Lag factors by 1 week
    df_model = df.copy()
    df_model["Bullish_lag"] = df_model["Bullish"].shift(1)
    df_model["Bearish_lag"] = df_model["Bearish"].shift(1)
    df_model = df_model.dropna()

    X = df_model[["Bullish_lag", "Bearish_lag"]]
    X = sm.add_constant(X)
    y = df_model["SP500_Return"]
    model = sm.OLS(y, X).fit()

    # Summary
    st.text("Regression Summary")
    st.text(model.summary())

    # Bar plot of coefficients
    st.subheader("Estimated Factor Loadings (β)")
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    betas = model.params[1:]
    betas.plot(kind="bar", color=["green", "red"], ax=ax3)
    ax3.set_ylabel("Coefficient")
    st.pyplot(fig3)

    # R² and residual diagnostics
    st.markdown(r"""
    **\( R^2 \) (Explained Variance)**: {:.3f}

    The \( R^2 \) value indicates the portion of return variance explained by sentiment factors (systematic risk). Remaining variation is attributed to residual (idiosyncratic) noise.
    """.format(model.rsquared))

    # Actual vs predicted
    st.markdown("**Actual vs Predicted Returns**")
    predicted = model.predict(X)
    fig4, ax4 = plt.subplots(figsize=(10, 3))
    ax4.plot(df_model.index, y, color="black", label="Actual", linewidth=0.6)
    ax4.plot(df_model.index, predicted, color="orange", linestyle="--", label="Predicted", linewidth=1)
    ax4.set_title("Actual vs Predicted S&P 500 Weekly Returns")
    ax4.set_ylabel("Weekly Return (%)")
    ax4.legend()
    st.pyplot(fig4)

    # Residual distribution
    st.markdown("**Distribution of Residuals**")
    residuals = y - predicted
    fig5, ax5 = plt.subplots(figsize=(6, 3))
    ax5.hist(residuals, bins=50, color="gray", edgecolor="black")
    ax5.set_title("Distribution of Residuals")
    st.pyplot(fig5)
