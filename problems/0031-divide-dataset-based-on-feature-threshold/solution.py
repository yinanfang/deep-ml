import numpy as np

def divide_on_feature(X, feature_i, threshold):
    return [[X[X[:, feature_i] >= threshold]], [X[X[:, feature_i] < threshold]]]
