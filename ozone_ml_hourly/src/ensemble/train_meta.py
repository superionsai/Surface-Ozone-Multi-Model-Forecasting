import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import glob
import pandas as pd
import numpy as np
import joblib
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

from src.features.policy import prepare_xy_xgb, prepare_xy_lstm
from src.models.lstm_model import AdvancedOzoneLSTM

def get_xgb_predictions(df_clean, station_name, indices):
    """Generates predictions for explicitly provided indices using the trained XGB model."""
    from src.features.policy import prepare_xy_xgb
    X, _ = prepare_xy_xgb(df_clean)
    
    X_target = X.iloc[indices]
    y_target = df_clean.iloc[indices]['Ozone (\u00b5g/m\u00b3)']
    baseline = df_clean.iloc[indices]['Ozone_Baseline']
    timestamps = df_clean.iloc[indices]['Timestamp']
    
    # Ensure features are in the exact same order as used during training
    feature_names = X.columns.tolist()
    X_target = X_target[feature_names]
    
    model_dir = f'models/xgb/{station_name}'
    if not os.path.exists(os.path.join(model_dir, 'xgb_advanced_best.json')):
        return None
        
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(model_dir, 'xgb_advanced_best.json'))
    feat_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
    target_scaler = joblib.load(os.path.join(model_dir, 'target_scaler.pkl'))
    
    X_scaled = feat_scaler.transform(X_target)
    preds_scaled = model.predict(X_scaled)
    preds_unscaled = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    preds_unscaled += baseline.values
    
    return pd.DataFrame({
        'Timestamp': pd.to_datetime(timestamps.values),
        'Actual': y_target.values,
        'Pred_XGB': np.clip(preds_unscaled, a_min=0, a_max=None)
    })

def get_lstm_predictions(df_clean, station_name, indices, seq_length=24):
    """Generates predictions for explicitly provided indices using the trained LSTM model."""
    model_dir = f'models/lstm/{station_name}'
    if not os.path.exists(os.path.join(model_dir, 'lstm_advanced_best.pth')):
        return None
        
    feat_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
    target_scaler = joblib.load(os.path.join(model_dir, 'target_scaler.pkl'))
    
    # Use the policy to get the exact features the LSTM expects
    X_all, _ = prepare_xy_lstm(df_clean)
    feature_names = X_all.columns.tolist()
    X_scaled = feat_scaler.transform(X_all[feature_names].values)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedOzoneLSTM(input_size=len(feature_names), hidden_size=64, num_layers=2, dropout=0.4)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'lstm_advanced_best.pth'), map_location=device))
    model.to(device)
    model.eval()
    
    preds_scaled = []
    with torch.no_grad():
        for idx in indices:
            start_idx = idx - seq_length
            seq = X_scaled[start_idx:start_idx + seq_length]
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            preds_scaled.append(model(seq_tensor).item())
            
    preds_unscaled = target_scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    baseline = df_clean.iloc[indices]['Ozone_Baseline']
    preds_unscaled += baseline.values
    
    return pd.DataFrame({
        'Timestamp': pd.to_datetime(df_clean.iloc[indices]['Timestamp'].values),
        'Pred_LSTM': np.clip(preds_unscaled, a_min=0, a_max=None)
    })

def build_unified_dataset(xgb_df, lstm_df):
    """Merges predictions on timestamp to ensure exact alignment."""
    unified_df = pd.merge(xgb_df, lstm_df, on='Timestamp', how='inner')
    
    # Feature Engineering for the Meta-Learner
    unified_df['Hour'] = unified_df['Timestamp'].dt.hour
    unified_df['Month'] = unified_df['Timestamp'].dt.month
    unified_df['DayOfWeek'] = unified_df['Timestamp'].dt.dayofweek
    
    # Cyclic encodings for time
    unified_df['Hour_Sin'] = np.sin(2 * np.pi * unified_df['Hour'] / 24.0)
    unified_df['Hour_Cos'] = np.cos(2 * np.pi * unified_df['Hour'] / 24.0)
    
    return unified_df

def train_station_meta_learner(station_file, output_dir):
    station_name = os.path.basename(station_file).replace('_cleaned.csv', '')
    print(f"\n========== Training Meta-Learner for {station_name} ==========")
    df = pd.read_csv(station_file)
    df_clean = df.interpolate(method='linear', limit_direction='both').ffill().bfill()
    
    from src.data.split_logic import get_global_splits
    train_idx, val_idx, test_idx = get_global_splits(df_clean, seq_length=24)
    
    # Meta-Learner trains EXCLUSIVELY on Validation data because base models never trained on it
    # This prevents the Meta-Learner from trusting overfit base model predictions
    xgb_df = get_xgb_predictions(df_clean, station_name, val_idx)
    lstm_df = get_lstm_predictions(df_clean, station_name, val_idx)
    
    if xgb_df is None or lstm_df is None:
        print(f"Skipping {station_name}: Missing base models.")
        return
        
    unified_df = build_unified_dataset(xgb_df, lstm_df)
    print(f"Meta-Train Size (Base Validation Set): {len(unified_df)} perfectly aligned timestamps.")
    
    features = ['Pred_XGB', 'Pred_LSTM', 'Hour_Sin', 'Hour_Cos', 'Month', 'DayOfWeek']
    target = 'Actual'
    
    X_train, y_train = unified_df[features], unified_df[target]
    
    # Random Forest with strong regularization to avoid overfitting the validation set
    from sklearn.model_selection import RandomizedSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [20, 50]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', random_state=42)
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    print(f"Best Meta-Learner Params: {search.best_params_}")
    print(f"Cross-Val RMSE: {-search.best_score_:.4f}")
    
    # Save Model (Only trained on the strict Meta-Train set to ensure remaining data is completely invisible)
    model_dir = os.path.join(output_dir, station_name)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, "meta_learner.pkl"))
    
    print(f"Saved Meta-Learner to {model_dir}")

def main():
    processed_dir = 'data/processed'
    output_dir = 'models/ensemble/meta_learner'
    
    os.makedirs(output_dir, exist_ok=True)
    station_files = glob.glob(os.path.join(processed_dir, "*_cleaned.csv"))
    
    for file in station_files:
        train_station_meta_learner(file, output_dir)

if __name__ == '__main__':
    main()
