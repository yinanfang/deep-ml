import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    scores = np.array(scores)
    total = sum(np.exp(scores))
    # print(total)
    return np.round(np.exp(scores) / total, 4)