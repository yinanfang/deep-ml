# learning: np.zeros() + np.arange()
import numpy as np

def to_categorical(x, n_col=None):
    rows = x.shape[0]
    cols = n_col if n_col else np.amax(x) + 1
    one_hot = np.zeros((rows, cols))
    one_hot[np.arange(rows), x] = 1
    return one_hot

# x = np.array([0, 1, 2, 1, 0])
# print(x.shape)
# output = to_categorical(x)
# print(output)