import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import ShuffleSplit, ParameterSampler, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Ensure the paths are correct relative to the script execution
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.features.policy import prepare_xy_xgb
from src.models.xgb_model import get_xgb_base_model

def get_domain_constraints(feature_names):
    """
    Returns a tuple of monotonic constraints mapped to the order of features.
    1 = increasing, -1 = decreasing, 0 = no constraint.
    """
    constraints = []
    for feat in feature_names:
        # Higher temperature and solar radiation generally increase photochemical ozone production
        if 'AT' in feat or 'SR' in feat:
            constraints.append(1)
        else:
            constraints.append(0)
    return tuple(constraints)

def train_station_xgb_advanced(station_file, output_dir):
    station_name = os.path.basename(station_file).replace('_cleaned.csv', '')
    print(f"\n========== Training Advanced XGBoost for {station_name} ==========")
    
    # 1. Load data
    df = pd.read_csv(station_file)
    df_clean = df.interpolate(method='linear', limit_direction='both').ffill().bfill()
    
    # 2. Extract features avoiding target leakage
    X, y = prepare_xy_xgb(df_clean)
    
    # 3. Get Global Fixed Indices
    from src.data.split_logic import get_global_splits
    train_idx, val_idx, test_idx = get_global_splits(df_clean, seq_length=24)
    
    # 4. Extract sets based on global indices
    X_train_raw, y_train_raw = X.iloc[train_idx], y.iloc[train_idx]
    X_val_raw, y_val_raw = X.iloc[val_idx], y.iloc[val_idx]
    
    # 5. Setup Scalers (Fit ONLY on train set)
    feat_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    feat_scaler.fit(X_train_raw)
    target_scaler.fit(y_train_raw.values.reshape(-1, 1))
    
    X_train_scaled = pd.DataFrame(feat_scaler.transform(X_train_raw), columns=X.columns)
    y_train_scaled = pd.Series(target_scaler.transform(y_train_raw.values.reshape(-1, 1)).flatten())
    
    X_val_scaled = pd.DataFrame(feat_scaler.transform(X_val_raw), columns=X.columns)
    y_val_scaled = pd.Series(target_scaler.transform(y_val_raw.values.reshape(-1, 1)).flatten())
    
    # Generate domain constraints dynamically based on available columns
    constraints = get_domain_constraints(X_train_scaled.columns.tolist())
    
    # 6. Setup High-Performance Randomized Search
    from sklearn.model_selection import ParameterSampler
    param_grid = {
        'max_depth': [12, 15, 18],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 5, 10]
    }
    
    # Sample 15 random combinations for speed
    n_iter = 15
    param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))
    num_total = len(param_list)
    
    best_score = float('inf')
    best_params = None
    best_model = None
    
    print(f"Executing Fixed Validation Search ({num_total} iterations)...")
    for i, params in enumerate(param_list):
        model = get_xgb_base_model(monotone_constraints=constraints)
        model.set_params(**params)
        
        # Use the validation fold directly for early stopping
        model.fit(X_train_scaled, y_train_scaled, eval_set=[(X_val_scaled, y_val_scaled)], verbose=False)
        
        preds = model.predict(X_val_scaled)
        mse = mean_squared_error(y_val_scaled, preds)
        
        if i % 5 == 0 or mse < best_score:
            print(f"Iter {i+1}/{num_total} | Val MSE (Scaled): {mse:.4f} | Depth: {params['max_depth']} | LR: {params['learning_rate']}")
        
        if mse < best_score:
            best_score = mse
            best_params = params
            best_model = model
            
    print(f"\nBest Val MSE (Scaled): {best_score:.4f}")
    
    # 6. Save Model and Scalers
    model_dir = os.path.join(output_dir, station_name)
    os.makedirs(model_dir, exist_ok=True)
    
    best_model.save_model(os.path.join(model_dir, "xgb_advanced_best.json"))
    joblib.dump(feat_scaler, os.path.join(model_dir, "feature_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(model_dir, "target_scaler.pkl"))
    
    print(f"Saved advanced XGBoost model and scalers to {model_dir}")

def main():
    processed_dir = 'data/processed'
    output_dir = 'models/xgb'
    
    if not os.path.exists(processed_dir):
        print(f"Directory {processed_dir} not found. Run preprocessing first.")
        return
        
    station_files = glob.glob(os.path.join(processed_dir, "*_cleaned.csv"))
    
    for file in station_files:
        train_station_xgb_advanced(file, output_dir)

if __name__ == '__main__':
    main()
