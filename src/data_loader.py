import yfinance as yf
import pandas as pd

def load_market_data(ticker="^NSEI", start="2010-01-01", end=None):
    df = yf.download(ticker, start=start, end=end)

    # 🔥 FIX: Flatten columns if Yahoo returns multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 🔥 ENSURE 1D SERIES (no DataFrames)
    df["Open"]  = df["Open"].squeeze()
    df["High"]  = df["High"].squeeze()
    df["Low"]   = df["Low"].squeeze()
    df["Close"] = df["Close"].squeeze()

    df.dropna(inplace=True)
    return df
