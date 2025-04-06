import math
import numpy as np
from collections import defaultdict

def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
    centroids = np.array(initial_centroids)

    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    for _ in range(max_iterations):
        clusters = defaultdict(list)
        for p in points:
            closest = min(range(k), key=lambda i: euclidean_distance(p, centroids[i]))
            clusters[closest].append(p)
        
        new_centroids = []
        for i in range(k):
            cluster_points = np.array(clusters[i])
            if cluster_points.any():
                # print(cluster_points)
                # x_avg, y_avg = np.round(np.mean(cluster_points, axis=0), 4)
                new_centroids.append(np.round(np.mean(cluster_points, axis=0), 4))
            else:
                new_ce