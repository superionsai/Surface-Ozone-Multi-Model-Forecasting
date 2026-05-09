import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_base_metrics(y_true, y_pred):
    """Calculates standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

def calc_ia(y_true, y_pred):
    """
    Index of Agreement (IA).
    Ranges from 0 (complete disagreement) to 1 (perfect agreement).
    """
    mean_obs = np.mean(y_true)
    numerator = np.sum((y_pred - y_true)**2)
    denominator = np.sum((np.abs(y_pred - mean_obs) + np.abs(y_true - mean_obs))**2)
    
    if denominator == 0:
        return 0.0
    return 1 - (numerator / denominator)

def calc_nmb(y_true, y_pred):
    """Normalized Mean Bias (NMB)."""
    numerator = np.sum(y_pred - y_true)
    denominator = np.sum(y_true)
    if denominator == 0:
        return np.nan
    return (numerator / denominator) * 100

def calc_nme(y_true, y_pred):
    """Normalized Mean Error (NME)."""
    numerator = np.sum(np.abs(y_pred - y_true))
    denominator = np.sum(y_true)
    if denominator == 0:
        return np.nan
    return (numerator / denominator) * 100

def calc_pbias(y_true, y_pred):
    """Percent Bias (PBIAS). Similar to NMB but standard in hydrology/air-quality."""
    return calc_nmb(y_true, y_pred)

def calc_fac2(y_true, y_pred):
    """Fraction of predictions within a factor of 2 of observations (FAC2)."""
    # Avoid division by zero
    mask = y_true > 0
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    if len(y_t) == 0:
        return 0.0
        
    ratio = y_p / y_t
    fac2_count = np.sum((ratio >= 0.5) & (ratio <= 2.0))
    return fac2_count / len(y_t)

def calc_ppa(y_true, y_pred, percentile=90):
    """
    Peak Prediction Accuracy (PPA).
    Calculates RMSE only for values above the given percentile of observations.
    """
    threshold = np.percentile(y_true, percentile)
    peak_mask = y_true >= threshold
    
    if np.sum(peak_mask) == 0:
        return np.nan
        
    y_t_peak = y_true[peak_mask]
    y_p_peak = y_pred[peak_mask]
    return np.sqrt(mean_squared_error(y_t_peak, y_p_peak))

def get_all_metrics(y_true, y_pred):
    """Returns a dictionary of all advanced metrics."""
    metrics = calculate_base_metrics(y_true, y_pred)
    metrics['IA'] = calc_ia(y_true, y_pred)
    metrics['NMB (%)'] = calc_nmb(y_true, y_pred)
    metrics['NME (%)'] = calc_nme(y_true, y_pred)
    metrics['PBIAS'] = calc_pbias(y_true, y_pred)
    metrics['FAC2'] = calc_fac2(y_true, y_pred)
    metrics['Peak RMSE (Top 10%)'] = calc_ppa(y_true, y_pred, percentile=90)
    return metrics
