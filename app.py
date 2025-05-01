import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# Load and clean the Excel data
file_path = "/mnt/data/aaii_mkt_sntmt.xlsx"
df = pd.read_excel(file_path, skiprows=4)
df = df.rename(columns={df.columns[0]: "Date"})

# Safely convert Date and drop bad rows
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df[df["Date"].notna()]  # Drop rows with NaT

# Ensure 'Bullish' is numeric for plotting and modeling
df = df[pd.to_numeric(df["Bullish"], errors="coerce").notna()]
df = df.sort_values("Date")

# Compute S&P 500 weekly returns
df["Return_SP500"] = df["S&P 500 Weekly Close"].pct_change() * 100

# Create lagged sentiment factors
df["Bullish_lag"] = df["Bullish"].shift(1)
df["Bearish_lag"] = df["Bearish"].shift(1)

# Drop NaNs from lagged values
df_model = df.dropna(subset=["Return_SP500", "Bullish_lag", "Bearish_lag"])

# Prepare regression variables
X = df_model[["Bullish_lag", "Bearish_lag"]]
X = sm.add_constant(X)
y = df_model["Return_SP500"]

# Fit OLS model
model = sm.OLS(y, X).fit()
df_model["Predicted_Return"] = model.predict(X)
df_model["Residual"] = df_model["Return_SP500"] - df_model["Predicted_Return"]

# Plot Actual vs Predicted
plt.figure(figsize=(10, 4))
plt.plot(df_model["Date"], df_model["Return_SP500"], label="Actual", color="black", linewidth=0.6)
plt.plot(df_model["Date"], df_model["Predicted_Return"], label="Predicted", color="orange", linestyle="--", linewidth=1.2)
plt.title("Actual vs Predicted S&P 500 Weekly Returns")
plt.ylabel("Weekly Return (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
