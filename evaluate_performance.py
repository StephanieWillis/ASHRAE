import numpy as np

def calculate_RMSLE_score(predictions, targets):
    n = len(predictions)
    rms_logs = (np.log(targets + 1) - np.log(predictions + 1))**2
    score = np.sqrt(1/n * rms_logs.sum())
    return score