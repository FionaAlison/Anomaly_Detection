import flwr as fl
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Tuple
import os
import datetime
from tensorflow.keras.callbacks import TensorBoard
from model import create_lstm_model


# Flower Client
class FLClient(fl.client.NumPyClient):
    def __init__(self, x: np.ndarray, y: np.ndarray, test_size: float = 0.2, k: int = 5, client_id: str = "Client"):
        """
        Args:
            x: Input features (num_samples, timesteps, features)
            y: Binary labels (num_samples,)
            test_size: Proportion of data to be used for testing
            k: Number of folds for cross-validation
            client_id: Identifier for the client (for logging purposes)
        """
        self.client_id = client_id
        self.x = x
        self.y = y
        self.test_size = test_size
        self.k = k
        self.input_shape = (x.shape[1], x.shape[2])  # (timesteps, features)

        # Split data into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=42
        )
        
        # Create the model
        self.model = create_lstm_model(self.input_shape)

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model parameters as NumPy arrays"""
        return self.model.get_weights()

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from NumPy arrays"""
        self.model.set_weights(parameters)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model using k-fold cross-validation and store mean metrics per epoch"""
        self.set_parameters(parameters)

        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        weights_list = []

        # Initialize containers for per-epoch metrics
        fold_accuracies = []
        fold_val_accuracies = []
        fold_losses = []
        fold_val_losses = []

        for fold, (train_index, val_index) in enumerate(kf.split(self.x_train)):
            fold_model = create_lstm_model(self.input_shape)
            fold_model.set_weights(parameters)

            history = fold_model.fit(
                self.x_train[train_index],
                self.y_train[train_index],
                validation_data=(self.x_train[val_index], self.y_train[val_index]),
                epochs=config.get("local_epochs", 3),
                batch_size=config.get("batch_size", 32),
                verbose=1
            )
            weights_list.append(fold_model.get_weights())

            # Collect per-epoch metrics
            fold_accuracies.append(history.history.get('accuracy', []))
            fold_val_accuracies.append(history.history.get('val_accuracy', []))
            fold_losses.append(history.history.get('loss', []))
            fold_val_losses.append(history.history.get('val_loss', []))

        # Compute mean per epoch across all folds
        mean_accuracy = np.mean(fold_accuracies, axis=0).tolist()
        mean_val_accuracy = np.mean(fold_val_accuracies, axis=0).tolist()
        mean_loss = np.mean(fold_losses, axis=0).tolist()
        mean_val_loss = np.mean(fold_val_losses, axis=0).tolist()

        new_weights = [
            np.mean([w[i] for w in weights_list], axis=0)
            for i in range(len(weights_list[0]))
        ]
        self.model.set_weights(new_weights)

        # Save mean metrics across all folds
        self.metrics = {
            'mean_accuracy': mean_accuracy,
            'mean_val_accuracy': mean_val_accuracy,
            'mean_loss': mean_loss,
            'mean_val_loss': mean_val_loss
        }

        return new_weights, len(self.x_train), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the model on the local test set"""
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return float(loss), len(self.x_test), {"accuracy": float(accuracy)}

def start_client(x: np.ndarray, y: np.ndarray, client_id: str) -> None:
    """Start Flower client with local data"""
    client = FLClient(x, y, client_id=client_id)
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()  # Convert to ClientApp compatible format
    )
