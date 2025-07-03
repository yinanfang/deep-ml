import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	_max = np.max(scores)
    log_sum = np.log(np.sum(np.exp(scores - _max)))
    return scores - _max - log_sum