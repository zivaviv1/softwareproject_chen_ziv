import sys

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

import symnmfmodule

MAX_ITER = 300
EPSILON = 1e-4


def load_points(path):
    return np.loadtxt(path, delimiter=",", dtype=float, ndmin=2)


def initialize_h(w, k):
    upper = 2 * np.sqrt(float(np.mean(w)) / k)
    return np.random.uniform(0, upper, size=(len(w), k))


def labels_for(points, centroids):
    distances = np.sum((points[:, None] - centroids) ** 2, axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(points, labels, centroids, k):
    updated = centroids.copy()
    for cluster in range(k):
        members = points[labels == cluster]
        if len(members) > 0:
            updated[cluster] = np.mean(members, axis=0)
    return updated


def kmeans_labels(points, k):
    centroids = points[:k].copy()
    for _ in range(MAX_ITER):
        labels = labels_for(points, centroids)
        updated = update_centroids(points, labels, centroids, k)
        if np.max(np.linalg.norm(updated - centroids, axis=1)) < EPSILON:
            centroids = updated
            break
        centroids = updated
    return labels_for(points, centroids)


def symnmf_labels(points, k):
    np.random.seed(1234)
    w = symnmfmodule.norm(points.tolist())
    h = initialize_h(w, k)
    result = symnmfmodule.symnmf(h.tolist(), w, MAX_ITER, EPSILON)
    return np.argmax(np.asarray(result), axis=1)


def main():
    k = int(sys.argv[1])
    points = load_points(sys.argv[2])
    nmf_labels = symnmf_labels(points, k)
    kmeans = kmeans_labels(points, k)
    print(f"nmf: {silhouette_score(points, nmf_labels):.4f}")
    print(f"kmeans: {silhouette_score(points, kmeans):.4f}")
    print(f"ari: {adjusted_rand_score(nmf_labels, kmeans):.4f}")


if __name__ == "__main__":
    main()
