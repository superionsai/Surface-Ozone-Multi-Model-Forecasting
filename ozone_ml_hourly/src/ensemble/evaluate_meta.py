import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import glob
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Re-use the data extractors we wrote for the training script
from src.ensemble.train_meta import get_xgb_predictions, get_lstm_predictions, build_unified_dataset
from src.evaluation.evaluate import generate_report
from src.evaluation.metrics import get_all_metrics

def evaluate_station_meta_learner(station_file, output_root):
    station_name = os.path.basename(station_file).replace('_cleaned.csv', '')
    model_dir = f'models/ensemble/meta_learner/{station_name}'
    output_dir = os.path.join(output_root, 'meta_learner', station_name)
    
    if not os.path.exists(os.path.join(model_dir, 'meta_learner.pkl')):
        print(f"Skipping Meta-Learner Evaluation {station_name}: Model not found.")
        return
        
    meta_learner = joblib.load(os.path.join(model_dir, 'meta_learner.pkl'))
    
    print(f"Evaluating Meta-Learner for {station_name}...")
    
    # 1. Load Data and get Global Split
    df = pd.read_csv(station_file)
    df_clean = df.interpolate(method='linear', limit_direction='both').ffill().bfill()
    
    from src.data.split_logic import get_global_splits
    train_idx, val_idx, test_idx = get_global_splits(df_clean, seq_length=24)
    
    # 2. Extract Base Predictions ONLY for the unseen test set
    xgb_df = get_xgb_predictions(df_clean, station_name, test_idx)
    lstm_df = get_lstm_predictions(df_clean, station_name, test_idx)
    
    if xgb_df is None or lstm_df is None:
        print(f"Skipping {station_name}: Missing base models.")
        return
        
    unified_df = build_unified_dataset(xgb_df, lstm_df)
    
    # Sort chronologically for line plots
    test_df = unified_df.sort_values('Timestamp').copy()
    
    # 3. Generate Meta-Learner Predictions
    features = ['Pred_XGB', 'Pred_LSTM', 'Hour_Sin', 'Hour_Cos', 'Month', 'DayOfWeek']
    test_df['Predicted'] = meta_learner.predict(test_df[features])
    
    # 4. Prepare DataFrame for generate_report (it expects 'Season', 'Hour', 'Month')
    def get_season(month):
        if month in [3, 4, 5, 6]: return 'Summer'
        elif month in [7, 8, 9]: return 'Monsoon'
        elif month in [10, 11]: return 'Post-Monsoon'
        else: return 'Winter'
    test_df['Season'] = test_df['Month'].apply(get_season)
    
    # 5. Calculate Metrics
    metrics = get_all_metrics(test_df['Actual'].values, test_df['Predicted'].values)
    all_metrics = {'Test': metrics}
    
    # 6. Generate the exhaustive Phase 4 report for the ensemble
    generate_report('meta_learner', station_name, test_df, output_dir, metrics=all_metrics)
    print(f"Finished evaluation for {station_name}.")

def main():
    processed_dir = 'data/processed'
    output_root = 'reports/figures/phase5_ensemble'
    
    station_files = glob.glob(os.path.join(processed_dir, "*_cleaned.csv"))
    
    for file in station_files:
        evaluate_station_meta_learner(file, output_root)

if __name__ == '__main__':
    main()
