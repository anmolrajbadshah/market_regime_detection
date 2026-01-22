ML-Powered Market Regime Detection & Strategy System

An end-to-end machine learning system that detects, predicts, and exploits market regimes (Bull, Bear, Sideways) using probabilistic models and deep learning.
The project integrates regime-aware strategy switching, realistic backtesting with transaction costs, risk analysis, and a cloud-deployed interactive dashboard.

Key Highlights

Unsupervised regime detection using Hidden Markov Models (HMMs)

Sequential regime prediction using LSTM networks

Regime-aware strategy switching (trend-following, mean-reversion, defensive)

Realistic backtesting engine with transaction costs & slippage

Risk analytics using equity curves, Sharpe ratio, and drawdown

Interactive Streamlit dashboard, deployed on Streamlit Cloud

Dockerized application for reproducible deployment

System Architecture

Market Data (OHLC)
        ↓
Feature Engineering
        ↓
HMM (Market Regime Detection)
        ↓
LSTM (Next Regime Prediction)
        ↓
Strategy Switching Engine
        ↓
Backtesting + Risk Analysis
        ↓
Streamlit Dashboard

The dashboard provides:

Current detected market regime

Active trading strategy & signal

Equity curve vs Buy-and-Hold

Drawdown visualization

Regime-wise performance breakdown

Models & Methods
🔹 Market Regime Detection

Hidden Markov Models (HMM) used to infer latent market states

Regimes mapped to Bull / Bear / Sideways based on return and volatility characteristics

🔹 Regime Prediction

LSTM network trained on engineered time-series features

Predicts the next market regime for forward-looking strategy decisions
🔹 Strategy Switching
Regime	Strategy
Bull	Trend-Following
Bear	Defensive / Cash
Sideways	Mean Reversion
🔹 Backtesting & Risk

Vectorized backtesting engine

Transaction costs and slippage applied on trades

Performance evaluated using:

Equity Curve

Sharpe Ratio

Maximum Drawdown

Regime-wise PnL attribution

Tech Stack

Language: Python

Data & Math: Pandas, NumPy

ML Models: hmmlearn (HMM), TensorFlow (LSTM)

Technical Indicators: ta

Visualization: Matplotlib

Dashboard: Streamlit

Deployment: Streamlit Cloud, Docker

Version Control: Git, GitHub

Project Structure
market_regime_detection/
│
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Dependencies
├── Dockerfile              # Container configuration
│
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── hmm_model.py
│   ├── strategy_engine.py
│   ├── backtest_engine.py
│   └── __init__.py
│
├── notebooks/              # Experiments & analysis
└── README.mdHow to Run Locally
1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/market_regime_detection.git
cd market_regime_detection

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run app.py

🐳 Run with Docker (Optional)
docker build -t market-regime-app .
docker run -p 8501:8501 market-regime-app


Open: http://localhost:8501

⚠️ Disclaimer

This project is for educational and research purposes only.
It does not constitute financial or investment advice.

📌 Key Learnings

Market behavior is better modeled through regimes than direct price prediction

Realistic evaluation requires costs, slippage, and drawdown analysis

ML projects gain real value when deployed and visualized, not just modeled

