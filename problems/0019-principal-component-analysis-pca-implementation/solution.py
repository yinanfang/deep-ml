# read through, understood, and re-implemented...

import numpy as np 
def pca(data: np.ndarray, k: int) -> np.ndarray:
    data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    covariance_matrix = np.cov(data_standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, idx]
    principal_components = eigenvectors_sorted[:, :k]
    return np.round(principal_components, 4)