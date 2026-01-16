import numpy as np
import pandas as pd


def generate_positions(signals):
    """
    Convert signals to positions:
    BUY  -> 1
    SELL -> 0
    HOLD -> previous
    CASH -> 0
    """
    positions = []

    for i, signal in enumerate(signals):
        if signal == "BUY":
            positions.append(1)
        elif signal in ["SELL", "CASH"]:
            positions.append(0)
        else:  # HOLD
            positions.append(positions[-1] if i > 0 else 0)

    return positions


def backtest_strategy(
    df,
    transaction_cost=0.001,   # 0.1% per trade
    slippage=0.0005            # 0.05% slippage
):
    df = df.copy()

    # Market returns
    df["market_return"] = df["Close"].pct_change()
 # Positions
    df["position"] = generate_positions(df["signal"])

    # Position change (trade happens here)
    df["trade"] = df["position"].diff().abs()
    # Convert signals to positions
    df["position"] = generate_positions(df["signal"])

# Strategy returns (NO look-ahead)
    gross_return = df["position"].shift(1) * df["market_return"]

    # Transaction costs + slippage
    cost = df["trade"] * (transaction_cost + slippage)
    # Strategy returns
    df["strategy_return"] = df["position"].shift(1) * df["market_return"]

    # Equity curve
    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()

    df.dropna(inplace=True)
    return df


def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def max_drawdown(equity_curve):
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()
