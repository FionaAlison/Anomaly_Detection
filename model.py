# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from typing import Tuple

def create_lstm_model(input_shape: Tuple[int, int], num_classes: int = 1) -> Sequential:
    """Create and compile an LSTM model for binary classification."""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dense(num_classes, activation="sigmoid" if num_classes == 1 else "softmax")
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
