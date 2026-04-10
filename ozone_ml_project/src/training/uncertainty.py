import numpy as np

def estimate_uncertainty(residuals):
    std = np.std(residuals)
    return std