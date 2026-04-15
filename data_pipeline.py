import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from config import Config

class SWaTDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_sliding_windows(data, labels, seq_length):
    """
    Convert 2D tabular data to 3D temporal sliding windows (Samples, Seq_Len, Features)
    """
    # Downsample by taking every Nth window to save memory during testing
    # Especially since SWaT has 400k rows
    stride = 5  
    
    x, y = [], []
    for i in range(0, len(data) - seq_length + 1, stride):
        x.append(data[i:i+seq_length])
        y_window = labels[i:i+seq_length]
        y.append(1 if np.sum(y_window) > 0 else 0)
    return np.array(x), np.array(y)

def apply_stage_features(df, stage_id):
    """
    Extracts features corresponding to a given stage (P1 to P6).
    In SWaT, feature names typically have numbers indicating their stage.
    Stage 1: *1##
    Stage 2: *2##
    etc.
    """
    # Clean column names by stripping spaces
    df.columns = df.columns.str.strip()
    
    stage_num = stage_id + 1
    # Match strings containing digits starting with stage_num
    # e.g. stage_num=1 -> FIT101, LIT101
    cols = [c for c in df.columns if any(char.isdigit() and str(stage_num) == char for char in c)]
    
    # Exclude Timestamp and Normal/Attack if they got matched accidentally
    cols = [c for c in cols if c not in ['Timestamp', 'Normal/Attack']]
    
    return df[cols], cols

def get_stage_dataloaders(stage_id):
    """
    Prepares train and test DataLoaders for a specific federated client using the real SWaT dataset.
    """
    # Load actual SWaT data
    train_path = 'dataset/normal.csv'
    test_path = 'dataset/attack.csv'
    
    # We only read a subset to keep simulation fast unless we want the full 400k rows locally
    # Config.TRAIN_SAMPLES_PER_STAGE allows us to cap it
    train_df = pd.read_csv(train_path, nrows=Config.TRAIN_SAMPLES_PER_STAGE)
    test_df = pd.read_csv(test_path, nrows=Config.TEST_SAMPLES_PER_STAGE)
    
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    # Labels
    train_labels = (train_df['Normal/Attack'] != 'Normal').astype(int).values
    test_labels = (test_df['Normal/Attack'] != 'Normal').astype(int).values
    
    # Stage feature extraction
    train_stage_df, features = apply_stage_features(train_df, stage_id)
    test_stage_df, _ = apply_stage_features(test_df, stage_id)
    
    # Scaling
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_stage_df.values)
    test_scaled = scaler.transform(test_stage_df.values)
    
    # Replace NaNs if any exist in the real dataset
    train_scaled = np.nan_to_num(train_scaled)
    test_scaled = np.nan_to_num(test_scaled)
    
    # Pad to fixed size for FedAvg architectural symmetry
    pad_len = Config.NUM_FEATURES - train_scaled.shape[1]
    if pad_len > 0:
        train_scaled = np.pad(train_scaled, ((0,0), (0, pad_len)), 'constant')
        test_scaled = np.pad(test_scaled, ((0,0), (0, pad_len)), 'constant')
    elif pad_len < 0:
        train_scaled = train_scaled[:, :Config.NUM_FEATURES]
        test_scaled = test_scaled[:, :Config.NUM_FEATURES]
    
    # Sliding windows
    x_train, y_train = create_sliding_windows(train_scaled, train_labels, Config.SEQ_LENGTH)
    x_test, y_test = create_sliding_windows(test_scaled, test_labels, Config.SEQ_LENGTH)
    
    train_dataset = SWaTDataset(x_train, y_train)
    test_dataset = SWaTDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, x_test, y_test, features

if __name__ == "__main__":
    train_loader, test_loader, x_test, y_test, features = get_stage_dataloaders(0)
    print(f"Features for Stage 1: {features}")
    print(f"X_Train shape: {next(iter(train_loader))[0].shape}")
    print(f"X_Test shape (sliding windows): {x_test.shape}")
