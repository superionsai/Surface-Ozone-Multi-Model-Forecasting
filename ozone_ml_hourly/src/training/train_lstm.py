import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

# Ensure the paths are correct relative to the script execution
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.sequence_gen import get_dataloaders
from src.features.policy import prepare_xy_lstm
from src.models.lstm_model import AdvancedOzoneLSTM
from src.data.split_logic import get_global_splits

def train_station_lstm_advanced(station_file, output_dir, patience=4, epochs=30 ):
    station_name = os.path.basename(station_file).replace('_cleaned.csv', '')
    print(f"\n========== Training Advanced BiLSTM for {station_name} ==========")
    
    # 1. Load data
    df = pd.read_csv(station_file)
    
    # 2. Extract features avoiding target leakage
    X, y = prepare_xy_lstm(df)
    
    # Re-combine into a single dataframe for sequence generation
    df_clean = pd.concat([X, y], axis=1)
    
    # Advanced NaN Handling
    df_clean = df_clean.interpolate(method='linear', limit_direction='both')
    df_clean = df_clean.ffill().bfill()
    
    # 3. Create dataloaders using the globally synchronized indices
    try:
        train_idx, val_idx, test_idx = get_global_splits(df_clean, seq_length=24)
        
        train_loader, val_loader, feature_scaler, target_scaler = get_dataloaders(
            df=df_clean,
            target_col='Ozone_Anomaly',
            train_idx=train_idx,
            val_idx=val_idx,
            seq_length=24,
            batch_size=64
        )
    except ValueError as e:
        print(f"Skipping {station_name} due to error: {e}")
        return
        
    # Get input size from the first batch
    X_sample, _ = next(iter(train_loader))
    input_size = X_sample.shape[2]
    
    # 4. Initialize Model, Loss, Optimizer (with Weight Decay), Scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = AdvancedOzoneLSTM(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.4).to(device)
    criterion = nn.MSELoss()
    # Added Weight Decay (L2 regularization) to prevent overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Advanced: Reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 6. Training Loop with Early Stopping & Gradient Clipping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            
            # Advanced: Gradient Clipping to stabilize BiLSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
            
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_losses.append(loss.item())
                
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Step the LR scheduler based on validation loss
        scheduler.step(avg_val_loss)
        
        if epoch % 2 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch}/{epochs}] | LR: {current_lr:.6f} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")
            
        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val MSE: {best_val_loss:.4f}")
                break
                
    # 7. Save the best model and its corresponding scaler
    model_dir = os.path.join(output_dir, station_name)
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "lstm_advanced_best.pth")
    feat_scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
    target_scaler_path = os.path.join(model_dir, "target_scaler.pkl")
    
    if best_model_state is not None:
        torch.save(best_model_state, model_path)
    joblib.dump(feature_scaler, feat_scaler_path)
    joblib.dump(target_scaler, target_scaler_path)
    
    print(f"Saved advanced model and scalers to {model_dir}")

def main():
    processed_dir = 'data/processed'
    output_dir = 'models/lstm'
    
    if not os.path.exists(processed_dir):
        print(f"Directory {processed_dir} not found. Run preprocessing first.")
        return
        
    station_files = glob.glob(os.path.join(processed_dir, "*_cleaned.csv"))
    
    for file in station_files:
        train_station_lstm_advanced(file, output_dir, patience=6)

if __name__ == '__main__':
    main()
