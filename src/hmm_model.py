import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def train_hmm(features_df, n_states=3):
    features = features_df[
        [
            "log_return",
            "volatility_10",
            "volatility_30",
            "trend_strength",
            "rsi",
            "macd",
            "hl_range",
            "atr"
        ]
    ].values

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )

    hmm.fit(features_scaled)
    hidden_states = hmm.predict(features_scaled)

    return hidden_states, hmm, scaler
