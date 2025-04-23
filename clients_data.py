import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# === CONFIG ===
DATA_FILE = 'CICIoT2023/Benign.csv'  # filepath
OUTPUT_DIR = './data'
NUM_CLIENTS = 5
SEQUENCE_LENGTH = 10
RANDOM_SEED = 42
TEST_SPLIT_RATIO = 0.2  # Split off 20% for the global test set

def preprocess_data(df):
    df.replace([np.inf, -np.inf], 0, inplace=True)  # infinite values
    df.fillna(0, inplace=True)  # NaNs

    # Optional: Encode 'Protocol Type' if it's categorical
    if 'Protocol Type' in df.columns:
        if df['Protocol Type'].dtype == object or df['Protocol Type'].nunique() < 20:
            df['Protocol Type'] = df['Protocol Type'].astype('category').cat.codes

    numeric_cols = [
        'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate', 'fin_flag_number', 'syn_flag_number',
        'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
        'ack_count', 'syn_count', 'fin_count', 'rst_count', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
        'Tot size', 'IAT', 'Number', 'Variance'
    ]

    binary_cols = [
        'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP',
        'ICMP', 'IGMP', 'IPv', 'LLC'
    ]

    numeric_cols = [col for col in numeric_cols if col in df.columns]

    scaler = StandardScaler()
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)

    return df, scaler

def create_sequences(data, labels, seq_len):
    x_seq, y_seq = [], []
    for i in range(len(data) - seq_len + 1):
        x_seq.append(data[i:i + seq_len])
        y_seq.append(labels[i + seq_len - 1])
    return np.array(x_seq), np.array(y_seq)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🔄 Loading dataset...")
    df = pd.read_csv(DATA_FILE)

    print("✅ Filtering benign samples only...")
    if 'label' in df.columns:
        df = df[df['label'].str.lower() == 'benign']
    else:
        print("⚠️ No 'label' column found, assuming all data is benign")

    print("🧼 Preprocessing benign data...")
    df, scaler = preprocess_data(df)

    print("📊 Extracting features and labels...")
    X = df.drop(columns=['label'], errors='ignore').values
    y = np.zeros(X.shape[0])  # Label all benign samples as 0

    print(f"🔁 Creating sequences with {SEQUENCE_LENGTH} timesteps...")
    X_seq, y_seq = create_sequences(X, y, SEQUENCE_LENGTH)

    print("🔀 Shuffling sequences...")
    X_seq, y_seq = shuffle(X_seq, y_seq, random_state=RANDOM_SEED)

    print(f"✅ Final sequence shape: {X_seq.shape} (samples, timesteps={SEQUENCE_LENGTH}, features)")

    # Split off the global test set
    test_size = int(TEST_SPLIT_RATIO * len(X_seq))
    X_test, y_test = X_seq[:test_size], y_seq[:test_size]
    X_train, y_train = X_seq[test_size:], y_seq[test_size:]

    # Save the global test set for evaluation later
    np.save(os.path.join(OUTPUT_DIR, "global_X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "global_y_test.npy"), y_test)

    # Split training data into client datasets
    chunk_size = len(X_train) // NUM_CLIENTS
    for i in range(NUM_CLIENTS):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != NUM_CLIENTS - 1 else len(X_train)

        np.save(os.path.join(OUTPUT_DIR, f"client_{i}_X.npy"), X_train[start:end])
        np.save(os.path.join(OUTPUT_DIR, f"client_{i}_y.npy"), y_train[start:end])

        print(f"✅ Saved client_{i+1} with {end - start} samples")    

    print("🎉 All client data saved successfully.")
