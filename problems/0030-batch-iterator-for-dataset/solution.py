import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    cur, n = 0, X.shape[0]
    res = []
    while cur < n:
		part = []
        part.extend(X[cur:cur + batch_size])
        if y is not None:
            part.extend(y[cur:cur + batch_size])
        cur += batch_size
		res.extend(part)
    return res