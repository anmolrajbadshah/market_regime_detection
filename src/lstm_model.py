import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


def create_sequences(X, y, window=30):
    """
    X: (n_samples, n_features)
    y: (n_samples,)
    """
    X_seq, y_seq = [], []

    for i in range(window, len(X)):
        X_seq.append(X[i-window:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)


def build_lstm(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
