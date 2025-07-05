import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
    m = len(y)
    for _ in range(n_iterations):
        if method == 'batch':
            predictions = X @ weights
            errors = predictions - y
            gradient = 2 / m * (X.T @ errors)
            weights -= learning_rate * gradient
        elif method == 'stochastic':
            for i in range(m):
                prediction = X[i] @ weights
                error = prediction - y[i]
                gradient = 2 * X[i].T.dot(error)
                weights -= learning_rate * gradient
        elif method == 'mini_batch':
            for i in range(0, m, batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                predictions = batch_X @ weights
                errors = predictions - batch_y
                gradient = 2 / batch_size * (batch_X.T @ errors)
                weights -= learning_rate * gradien