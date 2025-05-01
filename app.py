# --- Tab 2: S&P + Sentiment Charts ---
with tab2:
    st.header("📆 Select Time Range for Analysis")

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

    # Use two columns to fit plots tighter
    st.subheader("📊 Market Overview")

    col1, col2 = st.columns(1)

    with col1:
        st.markdown("##### 📉 S&P 500 Weekly Close (Log Scale)")
        fig1, ax1 = plt.subplots(figsize=(10, 2.5))  # Smaller height
        ax1.plot(filtered_df["Date"], filtered_df["SP500_Close"], color="black")
        ax1.set_yscale("log")
        ax1.set_ylabel("Price")
        ax1.grid(True, linestyle="--", linewidth=0.5)
        st.pyplot(fig1)

    with col1:
        st.markdown("##### 🧠 Investor Sentiment")
        fig2, ax2 = plt.subplots(figsize=(10, 2.5))  # Smaller height
        ax2.plot(filtered_df["Date"], filtered_df["Bullish"], label="Bullish", color="green")
        ax2.plot(filtered_df["Date"], filtered_df["Neutral"], label="Neutral", color="gray")
        ax2.plot(filtered_df["Date"], filtered_df["Bearish"], label="Bearish", color="red")
        ax2.set_ylabel("Sentiment (%)")
        ax2.legend()
        ax2.grid(True, linestyle="--", linewidth=0.5)
        st.pyplot(fig2)

    st.subheader("📋 Filtered Data Table")
    st.dataframe(filtered_df, use_container_width=True, height=400)
