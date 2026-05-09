import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class OzoneSequenceDataset(Dataset):
    """
    PyTorch Dataset for generating sliding window sequences for LSTM.
    Takes a DataFrame, extracts sequences of length `seq_length` for X,
    and the corresponding target `y` at the end of the sequence.
    """
    def __init__(self, df: pd.DataFrame, target_col: str, seq_length: int = 24):
        self.seq_length = seq_length
        self.target_col = target_col
        
        # Ensure target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in DataFrame.")
            
        # We need to drop non-numeric columns like 'Timestamp' if they exist in X
        self.features = [col for col in df.columns if col not in ['Timestamp', target_col]]
        
        # Convert to numpy arrays for faster slicing
        self.X_data = df[self.features].values.astype(np.float32)
        self.y_data = df[target_col].values.astype(np.float32)
        
        if np.isnan(self.X_data).any() or np.isnan(self.y_data).any():
            raise ValueError("NaN values detected in DataFrame. Please ensure all missing values are imputed before creating the dataset.")
        
        # Valid indices (we need at least seq_length history to predict the next step)
        self.num_samples = len(df) - seq_length

    def __len__(self):
        return max(0, self.num_samples)

    def __getitem__(self, idx):
        # Sequence from idx to idx + seq_length (exclusive)
        # Target is at idx + seq_length
        X_seq = self.X_data[idx:idx + self.seq_length]
        y_val = self.y_data[idx + self.seq_length]
        
        return torch.tensor(X_seq), torch.tensor(y_val)

def get_dataloaders(df: pd.DataFrame, target_col: str, train_idx: list, val_idx: list, seq_length: int = 24, batch_size: int = 64):
    """
    Utility to create training and validation dataloaders using pre-defined exact indices.
    Fits StandardScalers ONLY on the training dataset to prevent target leakage.
    Returns (train_loader, val_loader, feature_scaler, target_scaler).
    """
    from torch.utils.data import DataLoader, Subset
    from sklearn.preprocessing import StandardScaler
    
    # Identify feature columns
    features = [col for col in df.columns if col not in ['Timestamp', target_col]]
    
    # Fit scalers only on training data
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    train_df_raw = df.iloc[train_idx]
    feature_scaler.fit(train_df_raw[features])
    target_scaler.fit(train_df_raw[[target_col]])
    
    # Transform entire dataset
    df_scaled = df.copy()
    df_scaled[features] = feature_scaler.transform(df[features])
    df_scaled[target_col] = target_scaler.transform(df[[target_col]])
    
    # Create the full dataset of sequences
    full_dataset = OzoneSequenceDataset(df_scaled, target_col, seq_length)
    
    # Convert target row indices to sequence start indices
    train_seq_idx = [i - seq_length for i in train_idx]
    val_seq_idx = [i - seq_length for i in val_idx]
    
    train_dataset = Subset(full_dataset, train_seq_idx)
    val_dataset = Subset(full_dataset, val_seq_idx)
    
    # Create DataLoaders (shuffle=True only scrambles batch order, not the split!)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, feature_scaler, target_scaler
