import xgboost as xgb

def get_xgb_base_model(monotone_constraints=None):
    """
    Returns the base XGBoost regressor configured for robustness.
    We use Pseudo-Huber loss to mitigate the effect of extreme ozone outlier spikes.
    """
    params = {
        'objective': 'reg:pseudohubererror',
        'n_estimators': 500,
        'early_stopping_rounds': 50,
        'learning_rate': 0.05,
        'max_depth': 12,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42
    }
    
    if monotone_constraints is not None:
        params['monotone_constraints'] = monotone_constraints
        
    return xgb.XGBRegressor(**params)
