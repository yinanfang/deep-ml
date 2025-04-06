# read and copied...
import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# standardization
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std_data = (data - mean) / std

    # min-max
    _min = np.min(data, axis=0)
    _max = np.max(data, axis=0)
    normalized = (data - _min) / (_max - _min)
    return np.round(std_data, 4), np.round(normalized, 4)