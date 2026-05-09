import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_global_splits(df_clean, seq_length=24):
    """
    Hybrid Split Logic:
    1. Test: Last 1 year (8760 hours) - Fixed Chronological.
    2. Train: 65% of total data - Randomly sampled from the first 5 years.
    3. Val: Remainder of the first 5 years - Randomly sampled.
    """
    n_total = len(df_clean)
    n_test = 8760 # 1 year of hourly data
    
    # Define the pool for Train/Val (the first 5 years)
    first_5_years_idx = np.arange(seq_length, n_total - n_test)
    
    # Define the Test set (the last year)
    test_idx = list(range(n_total - n_test, n_total))
    
    # Calculate how many samples we need for 65% of TOTAL data
    n_train_target = int(n_total * 0.65)
    
    # Perform random split on the first 5 years
    # We use a fixed random_state for reproducibility across all models
    train_idx, val_idx = train_test_split(
        first_5_years_idx, 
        train_size=n_train_target, 
        random_state=42, 
        shuffle=True
    )
    
    return sorted(train_idx.tolist()), sorted(val_idx.tolist()), sorted(test_idx)
