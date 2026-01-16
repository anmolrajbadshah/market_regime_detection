import numpy as np
import pandas as pd
import ta

def create_features(df):
    df = df.copy()

    # FORCE TRUE 1D ARRAYS (NO EXCEPTIONS)
    close = df["Close"].to_numpy().ravel()
    high  = df["High"].to_numpy().ravel()
    low   = df["Low"].to_numpy().ravel()

    # Wrap back into Series with index
    close = pd.Series(close, index=df.index)
    high  = pd.Series(high, index=df.index)
    low   = pd.Series(low, index=df.index)

    # Returns
    df["log_return"] = np.log(close / close.shift(1))

    # Volatility
    df["volatility_10"] = df["log_return"].rolling(10).std()
    df["volatility_30"] = df["log_return"].rolling(30).std()

    # Trend
    df["ma_20"] = close.rolling(20).mean()
    df["ma_50"] = close.rolling(50).mean()
    df["trend_strength"] = df["ma_20"] - df["ma_50"]

    # Momentum
    df["rsi"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["macd"] = ta.trend.MACD(close=close).macd()

    # Range / volatility
    df["hl_range"] = (high - low) / close
    df["atr"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    df.dropna(inplace=True)
    return df
