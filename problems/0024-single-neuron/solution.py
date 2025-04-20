from typing import List, Tuple

import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> Tuple[List[float], float]:
    features, weights, labels = np.array(features), np.array(weights), np.array(labels)
    z = features @ weights + bias
    # print(z)
    y_hat = 1 / (1 + np.exp(-z))
    # print(y)
    loss = 1 / len(y_hat) * np.sum((y_hat - labels) ** 2)
    return np.round(y_hat, 4), np.round(loss, 4)    