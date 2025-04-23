import numpy as np

def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    B, C = np.array(B), np.array(C)
    return np.linalg.inv(C) @ B