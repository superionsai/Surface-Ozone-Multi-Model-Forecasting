import os
import sys
# Ensure the project root is in the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import glob
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np 
from src.features.policy import add_chemical_ratios, add_cyclical_time_features, add_meteorological_vectors

# Create processed directory if it doesn't exist
PROCESSED_DIR = 'data/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Important columns
TARGET_COL = 'Ozone (\u00b5g/m\u00b3)'
CRITICAL_FEAT = 'NO2 (\u00b5g/m\u00b3)'
MEDIUM_FEAT_SUBSTRINGS = ['AT', 'RH', 'WS', 'WD', 'SR', 'BP', 'NO (\u00b5g/m\u00b3)']

def get_medium_importance_columns(df):
    medium_cols = []
    for col in df.columns:
        if any(sub in col for sub in MEDIUM_FEAT_SUBSTRINGS):
            medium_cols.append(col)
    return medium_cols

def preprocess_station_data(file_paths, station_name):
    print(f"--- Processing Station: {station_name} ---")
    
    dfs = []
    for f in file_paths:
        try:
            df_part = pd.read_csv(f)
            dfs.append(df_part)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        print(f"No valid data found for {station_name}")
        return None
        
    # Concatenate all years for the station
    df = pd.concat(dfs, ignore_index=True)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])
        df = df.drop_duplicates(subset=['Timestamp'])
        df = df.sort_values('Timestamp').reset_index(drop=True)
    
    # Drop columns that are completely empty or have >95% missing values
    threshold = int(len(df) * 0.05)
    df = df.dropna(axis=1, thresh=threshold)
    print(f"Columns remaining after dropping mostly empty ones: {df.shape[1]}")
    
    # Handle medium importance interpolations
    medium_cols = get_medium_importance_columns(df)
    for col in medium_cols:
        if col in df.columns:
            # Linear interpolation up to 12 hours max
            df[col] = df[col].interpolate(method='linear', limit=12, limit_direction='both')
            
    # Handle critical NO2 and other features with Diurnal Mean Filling
    # This ensures a 100% continuous timeline for LSTM sequences
    print("Applying Diurnal Mean Filling to preserve continuous sequence for LSTM...")
    df['Hour'] = df['Timestamp'].dt.hour
    
    all_cols_to_fix = [TARGET_COL, CRITICAL_FEAT] + medium_cols
    for col in all_cols_to_fix:
        if col in df.columns:
            # First, do a small linear interpolation for 2-hour gaps
            df[col] = df[col].interpolate(method='linear', limit=2, limit_direction='both')
            
            # Second, fill larger gaps with the mean for that specific hour of the day
            diurnal_means = df.groupby('Hour')[col].transform('mean')
            df[col] = df[col].fillna(diurnal_means)
            
            # Last resort: global mean if the whole hour was missing
            df[col] = df[col].fillna(df[col].mean())
    
    df = df.drop(columns=['Hour'])
    print(f"Timeline preserved. Total rows kept: {len(df)}")
        
    # --- Advanced Feature Engineering ---
    print(f"Applying advanced feature engineering...")
    df = add_chemical_ratios(df)
    df = add_cyclical_time_features(df)
    df = add_meteorological_vectors(df)
    print(f"Features after engineering: {df.shape[1]}")
        
    # --- NEW: Clip the maximum deviating 5% of the target data (Winsorization) ---
    # We clip instead of dropping to maintain 100% sequence continuity for LSTM
    # --- NEW: Clip the top 5% of the target data (Winsorization) ---
    # Most Ozone outliers are unphysical spikes at the upper end.
    if TARGET_COL in df.columns:
        p95 = df[TARGET_COL].quantile(0.95)
        print(f"Capping upper 5% outliers for {TARGET_COL} at {p95:.2f}...")
        df[TARGET_COL] = df[TARGET_COL].clip(upper=p95)
        
        # --- NEW: Additive Decomposition (Rolling Baseline) ---
        # Calculate a 30-day (720 hours) strictly backward-looking baseline to avoid leakage.
        # Shift(1) ensures the current hour's target is not included in its own baseline.
        print("Calculating 30-day rolling baseline for target deseasonalization...")
        df['Ozone_Baseline'] = df[TARGET_COL].shift(1).rolling(window=720, min_periods=1).mean().bfill()
        
        # The new target for the models will be the anomaly
        df['Ozone_Anomaly'] = df[TARGET_COL] - df['Ozone_Baseline']
        
    # Save the processed data
    out_path = os.path.join(PROCESSED_DIR, f"{station_name}_cleaned.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {station_name} data to {out_path} with shape {df.shape}\n")
    return df

def main():
    base_dir = 'hourly_data'
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist.")
        return

    # List stations
    stations = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for station in stations:
        station_dir = os.path.join(base_dir, station)
        csv_files = glob.glob(os.path.join(station_dir, "*.csv"))
        if csv_files:
            preprocess_station_data(csv_files, station)

if __name__ == '__main__':
    main()
