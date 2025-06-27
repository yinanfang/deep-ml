import numpy as np

def make_diagonal(x):
	rows = cols = x.shape[0]
    m = np.zeros((rows, cols))
    m[np.arange(rows), np.arange(cols)] = x[np.arange(rows)]
    return m