
import numpy as np

def dice_score(y_true, y_pred):
    if np.all(y_true == 0) and np.all(y_pred == 0):
        return 0.0
    tp = np.dot(y_true, y_pred)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
	res = 2 * tp / (2 * tp + fp + fn)
	return round(res, 3)
