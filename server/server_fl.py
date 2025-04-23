import flwr as fl
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Precision, Recall

# === Load your global clean evaluation dataset here ===
data_dir = "./data"
X_test = np.load(os.path.join(data_dir, f"global_X_test.npy"))
y_test = np.load(os.path.join(data_dir, f"global_y_test.npy"))

# === Build your model ===
def get_model():
    from model import create_lstm_model   
    model = create_lstm_model(input_shape=X_test.shape[1:], num_classes=1)
    return model

# === Evaluation hook for the server ===
def evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config: Dict) -> Optional[Tuple[float, Dict]]:
    print(f"📊 Evaluating global model at round {server_round}...")

    model = get_model()
    model.set_weights(parameters)

    # Loss function and metrics
    loss_fn = SparseCategoricalCrossentropy(from_logits=False)
    acc_metric = SparseCategoricalAccuracy()
    precision_metric = Precision()
    recall_metric = Recall()

    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    loss = loss_fn(y_test, y_pred).numpy()
    acc_metric.update_state(y_test, y_pred)
    precision_metric.update_state(y_test, y_pred)
    recall_metric.update_state(y_test, y_pred)
    
    # Calculate metrics
    acc = acc_metric.result().numpy()
    precision = precision_metric.result().numpy()
    recall = recall_metric.result().numpy()
    
    # Calculate F1-Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    # Print the metrics
    print(f"✅ Global evaluation - Loss: {loss:.4f} | Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")
    
    return loss, {"accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1_score}

# === Config used per round ===
def get_on_fit_config_fn() -> callable:
    def fit_config(server_round: int) -> Dict:
        return {
            "round": server_round,
            "batch_size": 64,
            "local_epochs": 5,
        }
    return fit_config

# === Start the FL server ===
def start_server(num_rounds: int = 2, port: str = "8080"):
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=get_on_fit_config_fn(),
        evaluate_fn=evaluate_fn,  # Evaluation function for the server
    )

    fl.server.start_server(
        server_address=f"127.0.0.1:{port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
