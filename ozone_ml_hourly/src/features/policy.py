import pandas as pd
import numpy as np

def add_chemical_ratios(df: pd.DataFrame, epsilon: float = 1.0) -> pd.DataFrame:
    """
    Adds NO2_NO_ratio (Leighton ratio proxy) and VOC_NOx_ratio (Proxy using Benzene).
    """
    if 'NO2 (\u00b5g/m\u00b3)' in df.columns and 'NO (\u00b5g/m\u00b3)' in df.columns:
        df['NO2_NO_ratio'] = df['NO2 (\u00b5g/m\u00b3)'] / (df['NO (\u00b5g/m\u00b3)'] + epsilon)
        
    # VOC/NOx ratio proxy (using Benzene + Toluene if available)
    voc_cols = [col for col in ['Benzene (\u00b5g/m\u00b3)', 'Toluene (\u00b5g/m\u00b3)'] if col in df.columns]
    if voc_cols and 'NOx (ppb)' in df.columns:
        df['VOC_NOx_ratio'] = df[voc_cols].sum(axis=1, min_count=1) / (df['NOx (ppb)'] + epsilon)
        
    return df

def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts timestamps into continuous cyclical features (sin/cos).
    """
    if 'Timestamp' in df.columns:
        # Convert to datetime if it's not already
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Hour of day (0-23)
        hours = df['Timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
        
        # Month of year (1-12)
        months = df['Timestamp'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * months / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * months / 12.0)
        
    return df

def add_meteorological_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts Wind Speed (WS) and Wind Direction (WD) into U and V vectors.
    """
    if 'WS (m/s)' in df.columns and 'WD (deg)' in df.columns:
        # Convert degrees to radians
        wd_rad = df['WD (deg)'] * np.pi / 180.0
        
        # U vector (East-West) and V vector (North-South)
        # Note: Meteorological convention means wind *from* direction
        df['Wind_U'] = -df['WS (m/s)'] * np.sin(wd_rad)
        df['Wind_V'] = -df['WS (m/s)'] * np.cos(wd_rad)
        
    return df

def prepare_xy_xgb(df: pd.DataFrame, target_col: str = 'Ozone_Anomaly'):
    """
    Prepares data for XGBoost model.
    Bans all O3 features and hand-built lags to prevent data leakage.
    Retains PM2.5 and PM10 as aerosol scattering predictors.
    """
    # Exclusions
    forbidden_substrings = ['Ozone', 'O3', 'lag', 'diff', 'rolling', 'volatility', 'interaction']
    
    features = []
    for col in df.columns:
        if col == target_col or col == 'Timestamp':
            continue
            
        # Bypass exclusion for PM features
        if 'PM2.5' in col or 'PM10' in col:
            features.append(col)
            continue
            
        if not any(sub.lower() in col.lower() for sub in forbidden_substrings):
            features.append(col)
            
    X = df[features]
    y = df[target_col] if target_col in df.columns else None
    
    return X, y

def prepare_xy_lstm(df: pd.DataFrame, target_col: str = 'Ozone_Anomaly'):
    """
    Prepares data for LSTM model.
    Bans O3-derived predictors but allows general sequences.
    """
    forbidden_substrings = ['Ozone', 'O3']
    
    features = []
    for col in df.columns:
        if col == target_col or col == 'Timestamp':
            continue
            
        # Bypass exclusion for PM features
        if 'PM2.5' in col or 'PM10' in col:
            features.append(col)
            continue
            
        if not any(sub.lower() in col.lower() for sub in forbidden_substrings):
            features.append(col)
            
    X = df[features]
    y = df[target_col] if target_col in df.columns else None
    
    return X, y
