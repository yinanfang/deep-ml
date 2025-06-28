# hard to get the transpose dimension part right. Need to practice more.

import numpy as np

def calculate_correlation_matrix(X, Y=None):
    def calculate_std_dev(A):
        return np.sqrt(np.mean((A - A.mean(0))**2, axis=0))
    if Y is None:
        Y = X
    n_sample = np.shape(X)[0]
    covariance = (1 / n_sample) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    std_x = np.expand_dims(calculate_std_dev(X), 1)
    std_y = np.expand_dims(calculate_std_dev(Y), 1)
    correlation_matrix = np.divide(covariance, std_x.dot(std_y.T))
    return correlation_matrix