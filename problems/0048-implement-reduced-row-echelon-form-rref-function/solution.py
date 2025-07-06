import numpy as np

def rref(matrix):
    # Convert to float for division operations
    A = matrix.astype(np.float32)
    n, m = A.shape
    row = 0

    for col in range(m):
        if row >= n:
            break

        none_zero_rel_id = np.nonzero(A[row:, col])[0]
        if len(none_zero_rel_id) == 0:
            continue

        k = none_zero_rel_id[0] + row
        A[[row, k]] = A[[k, row]]
        A[row] = A[row] / A[row, col]
        for j in range(n):
            if j != row:
                A[j] = A[j] - A[j, col] * A[row]
        row += 1

    return A
