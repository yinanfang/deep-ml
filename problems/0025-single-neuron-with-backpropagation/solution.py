from typing import List, Tuple
import numpy as np

def sigmoid(x: float) -> float:
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: float) -> float:
	return sigmoid(x) * (1 - sigmoid(x))

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> Tuple[np.ndarray, float, List[float]]:
    X = features
    y = labels
    w = initial_weights
    b = initial_bias
    n = len(y)
    losses = []
    for _ in range(epochs):
        z = X @ w.T + b
        a = sigmoid(z)
        loss = np.mean((a - y) ** 2)
        losses.append(loss)

        dz = 2 / n * (a - y) * sigmoid_derivative(z)
        dw = dz @ X
        db = np.sum(dz)
        
        w -= learning_rate * dw
        b -= learning_rate * db

    return np.round(w, 4), np.round(b, 4), np.round(losses, 4)