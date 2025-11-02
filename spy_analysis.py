import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path

# =============== SETTINGS ===============
TICKER = "SPY"
START_DATE = "2015-01-01"
OUT_DIR = Path.cwd() / "reports"
OUT_DIR.mkdir(exist_ok=True)
# ========================================

print("Downloading data...")
data = yf.download(TICKER, start=START_DATE)[["Close"]].rename(columns={"Close": "Price"})
data.dropna(inplace=True)

# ---- Daily returns
data["Return"] = data["Price"].pct_change()
returns = data["Return"].dropna()

# ---- Performance metrics
ann = 252
cagr = (data["Price"].iloc[-1] / data["Price"].iloc[0]) ** (ann / len(data)) - 1
volatility = returns.std() * np.sqrt(ann)
sharpe = (returns.mean() * ann) / volatility if volatility else np.nan

# ---- Max drawdown
cum = (1 + returns).cumprod()
roll_max = cum.cummax()
drawdown = cum / roll_max - 1
max_dd = drawdown.min()

# ---- Summary
summary = pd.Series({
    "CAGR": cagr,
    "Volatility": volatility,
    "Sharpe Ratio": sharpe,
    "Max Drawdown": max_dd
})
print("\n--- Performance Summary ---")
print(summary.apply(lambda x: round(float(x), 4)))

# ---- Save files
summary.to_frame("value").to_csv(OUT_DIR / "spy_summary.csv")
returns.to_csv(OUT_DIR / "spy_daily_returns.csv")

# ---- Plot equity curve
plt.figure(figsize=(10, 6))
cum.plot(title=f"{TICKER} Equity Curve (Buy & Hold)", ylabel="Growth of $1")
plt.tight_layout()
plt.savefig(OUT_DIR / "spy_equity_curve.png", dpi=150)
plt.show()
