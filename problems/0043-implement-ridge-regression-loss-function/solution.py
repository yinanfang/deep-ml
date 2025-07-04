import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	loss = 1 / X.shape[0] * np.sum(np.power(w.dot(X.T) - y_true, 2))
    loss += alpha * np.sum(np.power(w, 2))
    return np.round(loss, 3)
