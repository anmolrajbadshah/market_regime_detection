def trend_following_strategy(df):
    signals = []

    for i in range(len(df)):
        if df["Close"].iloc[i] > df["ma_50"].iloc[i]:
            signals.append("BUY")
        else:
            signals.append("SELL")

    return signals
def defensive_strategy(df):
    return ["CASH"] * len(df)
def mean_reversion_strategy(df):
    signals = []

    for i in range(len(df)):
        if df["rsi"].iloc[i] < 30:
            signals.append("BUY")
        elif df["rsi"].iloc[i] > 70:
            signals.append("SELL")
        else:
            signals.append("HOLD")

    return signals
def strategy_switcher(df, regime):
    if regime == "Bull":
        return trend_following_strategy(df)
    elif regime == "Bear":
        return defensive_strategy(df)
    elif regime == "Sideways":
        return mean_reversion_strategy(df)
    else:
        return ["HOLD"] * len(df)