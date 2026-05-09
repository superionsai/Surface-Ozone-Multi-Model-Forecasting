import os
import sys
# Ensure the paths are correct relative to the script execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import glob
import pandas as pd
import numpy as np
import joblib
import torch

from src.features.policy import prepare_xy_xgb, prepare_xy_lstm
from src.evaluation.metrics import get_all_metrics
from src.evaluation.plots import (
    plot_diurnal_cycle, plot_seasonal_boxplots, plot_high_res_dynamics,
    plot_pdf_overlay, plot_aqi_confusion, plot_taylor_diagram,
    plot_regression_scatter, plot_error_bar_chart, plot_monthly_trends, 
    plot_hazardous_events, plot_shap_summary, plot_feature_importance
)
from src.models.lstm_model import AdvancedOzoneLSTM
import xgboost as xgb

def load_xgb_eval_data(station_file):
    df = pd.read_csv(station_file)
    df_clean = df.interpolate(method='linear', limit_direction='both').ffill().bfill()
    X, _ = prepare_xy_xgb(df_clean)
    
    from src.data.split_logic import get_global_splits
    train_idx, val_idx, test_idx = get_global_splits(df_clean, seq_length=24)
    
    def extract_split(indices):
        X_split = X.iloc[indices]
        y_true = df_clean.iloc[indices]['Ozone (\u00b5g/m\u00b3)']
        baseline = df_clean.iloc[indices]['Ozone_Baseline']
        t_split = df_clean.iloc[indices]['Timestamp']
        
        sort_indices = np.argsort(t_split.values)
        X_split = X_split.iloc[sort_indices]
        y_true = y_true.iloc[sort_indices]
        baseline = baseline.iloc[sort_indices]
        t_split = t_split.iloc[sort_indices]
        
        return X_split, pd.DataFrame({
            'Timestamp': pd.to_datetime(t_split), 
            'Actual': y_true,
            'Baseline': baseline
        })
        
    X_train, df_train = extract_split(train_idx)
    X_val, df_val = extract_split(val_idx)
    X_test, df_test = extract_split(test_idx)
    
    return (X_train, df_train), (X_val, df_val), (X_test, df_test)

def load_lstm_eval_data(station_file, seq_length=24):
    df = pd.read_csv(station_file)
    df_clean = df.interpolate(method='linear', limit_direction='both').ffill().bfill()
    X, _ = prepare_xy_lstm(df_clean)
    
    from src.data.split_logic import get_global_splits
    train_idx, val_idx, test_idx = get_global_splits(df_clean, seq_length=seq_length)
    
    def extract_split(indices):
        indices = sorted(indices)
        y_true = df_clean.iloc[indices]['Ozone (\u00b5g/m\u00b3)']
        baseline = df_clean.iloc[indices]['Ozone_Baseline']
        t_split = df_clean.iloc[indices]['Timestamp']
        
        df_eval = pd.DataFrame({
            'Timestamp': pd.to_datetime(t_split), 
            'Actual': y_true.values.flatten(),
            'Baseline': baseline.values.flatten()
        })
        return indices, df_eval
        
    train_idx, df_train = extract_split(train_idx)
    val_idx, df_val = extract_split(val_idx)
    test_idx, df_test = extract_split(test_idx)
    
    return df_clean, (train_idx, df_train), (val_idx, df_val), (test_idx, df_test)

def generate_report(model_type, station_name, df_eval, output_dir, metrics, model=None, X_val_scaled=None, feature_names=None):
    """Generates all plots and the text report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Add time-derived columns for plotting
    df_eval['Hour'] = df_eval['Timestamp'].dt.hour
    df_eval['Month'] = df_eval['Timestamp'].dt.month
    
    def get_season(month):
        if month in [3, 4, 5, 6]: return 'Summer'
        elif month in [7, 8, 9]: return 'Monsoon'
        elif month in [10, 11]: return 'Post-Monsoon'
        else: return 'Winter'
        
    df_eval['Season'] = df_eval['Month'].apply(get_season)
    
    # 2. Generate Plots
    print(f"Generating plots for {station_name} ({model_type})...")
    plot_diurnal_cycle(df_eval, os.path.join(output_dir, 'diurnal_cycle.png'))
    plot_seasonal_boxplots(df_eval, os.path.join(output_dir, 'seasonal_boxplots.png'))
    plot_high_res_dynamics(df_eval, os.path.join(output_dir, 'week_dynamics.png'), days=7)
    plot_pdf_overlay(df_eval, os.path.join(output_dir, 'pdf_distribution.png'))
    plot_aqi_confusion(df_eval, os.path.join(output_dir, 'aqi_matrix.png'))
    plot_taylor_diagram(df_eval, os.path.join(output_dir, 'taylor_diagram.png'))
    
    plot_regression_scatter(df_eval, os.path.join(output_dir, 'regression_scatter.png'))
    plot_error_bar_chart(df_eval, os.path.join(output_dir, 'seasonal_metrics_bar.png'))
    plot_monthly_trends(df_eval, os.path.join(output_dir, 'monthly_trends.png'))
    plot_hazardous_events(df_eval, os.path.join(output_dir, 'hazardous_events.png'))
    
    if model_type == 'xgb' and model is not None and X_val_scaled is not None:
        plot_shap_summary(model, X_val_scaled, os.path.join(output_dir, 'shap_summary.png'))
        if feature_names is not None:
            plot_feature_importance(model, feature_names, os.path.join(output_dir, 'feature_importance.png'))
    
    # 3. Write Report
    from src.evaluation.report_generator import generate_dense_report
    report_path = os.path.join(output_dir, 'report_summary.txt')
    generate_dense_report(model_type, station_name, df_eval, metrics, report_path)

def evaluate_xgb(station_file, models_dir, output_root):
    station_name = os.path.basename(station_file).replace('_cleaned.csv', '')
    model_dir = os.path.join(models_dir, station_name)
    output_dir = os.path.join(output_root, 'xgb', station_name)
    
    if not os.path.exists(os.path.join(model_dir, 'xgb_advanced_best.json')):
        print(f"Skipping XGB {station_name}: Model not found.")
        return
        
    (X_train, df_train), (X_val, df_val), (X_test, df_test) = load_xgb_eval_data(station_file)
    
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(model_dir, 'xgb_advanced_best.json'))
    
    feat_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
    target_scaler = joblib.load(os.path.join(model_dir, 'target_scaler.pkl'))
    
    def predict_split(X_split, df_split):
        X_scaled = feat_scaler.transform(X_split)
        preds_scaled = model.predict(X_scaled)
        preds_unscaled = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        df_split['Predicted_Anomaly'] = preds_unscaled
        df_split['Predicted'] = df_split['Predicted_Anomaly'] + df_split['Baseline']
        df_split['Predicted'] = df_split['Predicted'].clip(lower=0)
        return df_split

    df_train = predict_split(X_train, df_train)
    df_val = predict_split(X_val, df_val)
    df_test = predict_split(X_test, df_test)
    
    all_metrics = {
        'Train': get_all_metrics(df_train['Actual'].values, df_train['Predicted'].values),
        'Val': get_all_metrics(df_val['Actual'].values, df_val['Predicted'].values),
        'Test': get_all_metrics(df_test['Actual'].values, df_test['Predicted'].values)
    }
    
    # We pass the scaled dataframe so SHAP can interpret the actual model inputs
    X_val_scaled = feat_scaler.transform(X_val)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    
    generate_report('xgb', station_name, df_test, output_dir, metrics=all_metrics, model=model, X_val_scaled=X_val_scaled_df, feature_names=X_val.columns.tolist())

def evaluate_lstm(station_file, models_dir, output_root):
    station_name = os.path.basename(station_file).replace('_cleaned.csv', '')
    model_dir = os.path.join(models_dir, station_name)
    output_dir = os.path.join(output_root, 'lstm', station_name)
    
    if not os.path.exists(os.path.join(model_dir, 'lstm_advanced_best.pth')):
        print(f"Skipping LSTM {station_name}: Model not found.")
        return
        
    df_clean, (train_idx, df_train), (val_idx, df_val), (test_idx, df_test) = load_lstm_eval_data(station_file)
    
    feat_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
    target_scaler = joblib.load(os.path.join(model_dir, 'target_scaler.pkl'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use the policy to get the exact features the LSTM expects
    X_all, _ = prepare_xy_lstm(df_clean)
    features = X_all.columns.tolist()
    input_size = len(features)
    
    model = AdvancedOzoneLSTM(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.4)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'lstm_advanced_best.pth'), map_location=device))
    model.to(device)
    model.eval()
    
    X_data_scaled = feat_scaler.transform(df_clean[features].values)
    
    def predict_split(indices, df_split):
        preds_scaled = []
        with torch.no_grad():
            for idx in indices:
                start_idx = idx - 24
                seq = X_data_scaled[start_idx : start_idx + 24]
                seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                preds_scaled.append(model(seq_tensor).item())
                
        preds_unscaled = target_scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        df_split['Predicted_Anomaly'] = preds_unscaled
        df_split['Predicted'] = df_split['Predicted_Anomaly'] + df_split['Baseline']
        df_split['Predicted'] = df_split['Predicted'].clip(lower=0)
        return df_split

    df_train = predict_split(train_idx, df_train)
    df_val = predict_split(val_idx, df_val)
    df_test = predict_split(test_idx, df_test)
    
    all_metrics = {
        'Train': get_all_metrics(df_train['Actual'].values, df_train['Predicted'].values),
        'Val': get_all_metrics(df_val['Actual'].values, df_val['Predicted'].values),
        'Test': get_all_metrics(df_test['Actual'].values, df_test['Predicted'].values)
    }
    
    generate_report('lstm', station_name, df_test, output_dir, metrics=all_metrics)

def main():
    processed_dir = 'data/processed'
    output_root = 'reports/figures/phase4_individual'
    
    station_files = glob.glob(os.path.join(processed_dir, "*_cleaned.csv"))
    
    for file in station_files:
        evaluate_xgb(file, 'models/xgb', output_root)
        evaluate_lstm(file, 'models/lstm', output_root)

if __name__ == '__main__':
    main()
