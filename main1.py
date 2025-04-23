import multiprocessing
import time
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import flwr as fl

from server.server_fl import start_server
from client.client_fl import FLClient


def load_client_data(client_id):
    """Load and verify client data with detailed checks"""
    data_dir = "./data"
    try:
        x = np.load(os.path.join(data_dir, f"client_{client_id}_X.npy"))
        y = np.load(os.path.join(data_dir, f"client_{client_id}_y.npy"))
        
        # Data validation
        assert x.shape[0] == y.shape[0], "X and y have different sample counts"
        assert len(x.shape) == 3, "X should be 3D (samples, timesteps, features)"
        assert len(y.shape) == 1, "y should be 1D"
        
        print(f"✅ Client {client_id} data loaded successfully")
        print(f"   Samples: {x.shape[0]}, Timesteps: {x.shape[1]}, Features: {x.shape[2]}")
        print(f"   Class distribution: {np.bincount(y.astype(int))}")
        
        return x, y
    except Exception as e:
        print(f"❌ Error loading client {client_id} data: {str(e)}")
        raise


def run_server():
    print("\nSERVER: Starting...", flush=True)
    try:
        start_server(num_rounds=2)
        print("SERVER: Completed successfully", flush=True)
    except Exception as e:
        print(f"SERVER: Failed - {str(e)}", flush=True, file=sys.stderr)
        raise


def run_client(client_id):
    """Client process with metric tracking and storage"""
    print(f"CLIENT {client_id}: Starting", flush=True)
    try:
        x, y = load_client_data(client_id)
        client = FLClient(x, y)
        fl.client.start_client(
            server_address="127.0.0.1:8080",
            client=client.to_client()
        )

        # Save training metrics
        if hasattr(client, 'metrics'):
            with open(f"client_{client_id}_metrics.pkl", "wb") as f:
                pickle.dump(client.metrics, f)

        print(f"CLIENT {client_id}: Completed", flush=True)
    except Exception as e:
        print(f"CLIENT {client_id}: Failed - {str(e)}", flush=True, file=sys.stderr)


def plot_metrics():
    for i in range(5):  # Adjust based on the number of clients
        try:
            with open(f"client_{i}_metrics.pkl", "rb") as f:
                metrics = pickle.load(f)

                plt.figure(figsize=(12, 4))

                # Mean Accuracy
                plt.subplot(1, 3, 1)
                plt.plot(metrics.get("mean_accuracy", []), label="Mean Train Acc")
                plt.plot(metrics.get("mean_val_accuracy", []), label="Mean Val Acc")
                plt.title(f"Client {i} - Mean Accuracy")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.grid(True)

                # Mean Loss
                plt.subplot(1, 3, 2)
                plt.plot(metrics.get("mean_loss", []), label="Mean Train Loss")
                plt.plot(metrics.get("mean_val_loss", []), label="Mean Val Loss")
                plt.title(f"Client {i} - Mean Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True)

                # ASR (if available)
                plt.subplot(1, 3, 3)
                asr_values = metrics.get("asr", [])
                if asr_values:
                    plt.plot(asr_values, label="ASR", color="red")
                    plt.title(f"Client {i} - ASR")
                    plt.xlabel("Epoch")
                    plt.ylabel("ASR (%)")
                    plt.legend()
                    plt.grid(True)
                else:
                    plt.axis('off')
                    plt.text(0.3, 0.5, 'No ASR recorded', fontsize=12)

                plt.tight_layout()
                plt.savefig(f"client_{i}_metrics.png")
                plt.show()

        except FileNotFoundError:
            print(f"MAIN: Metrics for client {i} not found", flush=True)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    ctx = multiprocessing.get_context('spawn')

    # Start server
    server = ctx.Process(target=run_server, daemon=False)
    try:
        server.start()
        print("MAIN: Server process started", flush=True)
    except Exception as e:
        print(f"MAIN: Failed to start server process - {str(e)}", flush=True)
        sys.exit(1)

    time.sleep(5)  # Give server time
    # Start clients
    clients = []
    for i in range(5):
        p = ctx.Process(target=run_client, args=(i,), daemon=False)
        p.start()
        clients.append(p)
        print(f"MAIN: Started client {i}", flush=True)
        time.sleep(1)

    # Monitor processes
    try:
        while any(p.is_alive() for p in [server] + clients):
            time.sleep(1)
            server_status = "alive" if server.is_alive() else "stopped"
            client_status = sum(c.is_alive() for c in clients)
            print(f"MAIN: Status - Server: {server_status}, Active clients: {client_status}/{len(clients)}", flush=True)

    except KeyboardInterrupt:
        print("\nMAIN: Keyboard interrupt - shutting down", flush=True)

    finally:
        print("MAIN: Terminating processes...", flush=True)
        for p in clients:
            if p.is_alive():
                p.terminate()
        if server.is_alive():
            server.terminate()

        for p in clients:
            p.join()
        server.join()

        print("MAIN: All processes terminated", flush=True)

        # Plot results
        plot_metrics()
