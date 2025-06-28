# np.random logic was a bit complicated. X[idx][i] logics was a bit confusing. understood and copied...
import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
    np.random.seed(seed)
    n, m = X.shape
    subset_size = n if replacements else n // 2
    idx = np.array([np.random.choice(n, subset_size, replace=replacements) for _ in range(n_subsets)])
    return [(X[idx][i], y[idx][i]) for i in range(n_subsets)]

