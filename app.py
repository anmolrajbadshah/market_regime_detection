import matplotlib
matplotlib.use("Agg")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath("."))

from src.data_loader import load_market_data
from src.features import create_features
from src.hmm_model import train_hmm
from src.strategy_engine import strategy_switcher
from src.backtest_engine import backtest_strategy, sharpe_ratio, max_drawdown

st.set_page_config(page_title="Market Regime Dashboard", layout="wide")

st.title("📊 ML-Powered Market Regime Dashboard")

# -----------------------------
# Load and process data
# -----------------------------
with st.spinner("Loading market data..."):
    df = load_market_data()
    features_df = create_features(df)

    hidden_states, hmm_model, scaler = train_hmm(features_df)
    features_df["regime"] = hidden_states

regime_map = {
    0: "Bull",
    1: "Bear",
    2: "Sideways"
}

features_df["regime_name"] = features_df["regime"].map(regime_map)

# -----------------------------
# Apply strategy
# -----------------------------
features_df["signal"] = None

for regime in features_df["regime_name"].unique():
    mask = features_df["regime_name"] == regime
    features_df.loc[mask, "signal"] = strategy_switcher(
        features_df[mask],
        regime
    )

# -----------------------------
# Backtest
# -----------------------------
bt_df = backtest_strategy(
    features_df,
    transaction_cost=0.001,
    slippage=0.0005
)
# ---- Compute drawdown (FIX) ----
bt_df["rolling_max"] = bt_df["equity_curve"].cummax()
bt_df["drawdown"] = (bt_df["equity_curve"] - bt_df["rolling_max"]) / bt_df["rolling_max"]

# -----------------------------
# Sidebar summary
# -----------------------------
st.sidebar.header("📌 Current State")

current_regime = bt_df["regime_name"].iloc[-1]
current_signal = bt_df["signal"].iloc[-1]

st.sidebar.metric("Current Regime", current_regime)
st.sidebar.metric("Current Signal", current_signal)

sr = sharpe_ratio(bt_df["strategy_return"])
mdd = max_drawdown(bt_df["equity_curve"])

st.sidebar.metric("Sharpe Ratio", f"{sr:.2f}")
st.sidebar.metric("Max Drawdown", f"{mdd:.2%}")

# -----------------------------
# Equity curve
# -----------------------------
st.subheader("📈 Equity Curve")

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(bt_df.index, bt_df["equity_curve"], label="Strategy")
ax.plot(
    bt_df.index,
    (1 + bt_df["market_return"]).cumprod(),
    label="Buy & Hold",
    linestyle="--"
)
ax.legend()
st.pyplot(fig)

# -----------------------------
# Drawdown
# -----------------------------
st.subheader("📉 Drawdown")

fig, ax = plt.subplots(figsize=(10,3))
ax.fill_between(
    bt_df.index,
    bt_df["drawdown"],
    0,
    color="red",
    alpha=0.4
)
st.pyplot(fig)

# -----------------------------
# Regime performance table
# -----------------------------
st.subheader("📊 Regime-Wise Performance")

regime_perf = pd.DataFrame({
    "Annual Return": bt_df.groupby("regime_name")["strategy_return"].mean() * 252,
    "Sharpe Ratio": bt_df.groupby("regime_name")["strategy_return"].apply(sharpe_ratio),
    "Max Drawdown": bt_df.groupby("regime_name")["drawdown"].min()
})

st.dataframe(regime_perf.style.format({
    "Annual Return": "{:.2%}",
    "Sharpe Ratio": "{:.2f}",
    "Max Drawdown": "{:.2%}"
}))
