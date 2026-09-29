import sys
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
import symnmfmodule

np.random.seed(1234)

EPSILON = 1e-4
MAX_ITER = 300


def labels_for(points, centroids):
    distances = np.sum((points[:, None] - centroids) ** 2, axis=2)
    return np.argmin(distances, axis=1)


def kmeans_labels(points, k):
    centroids = points[:k].copy()

    for _ in range(MAX_ITER):
        labels = labels_for(points, centroids)
        updated = centroids.copy()

        for cluster in range(k):
            members = points[labels == cluster]
            if len(members) > 0:
                updated[cluster] = np.mean(members, axis=0)

        if np.max(np.linalg.norm(updated - centroids, axis=1)) < EPSILON:
            centroids = updated
            break

        centroids = updated

    return labels_for(points, centroids)

def _initialize_h(w, k):
    m = float(np.mean(w))
    return np.random.uniform(0, 2 * np.sqrt(m / k), size=(len(w), k))


def symnmf_labels(points, k):
    w = symnmfmodule.norm(points.tolist())
    h = _initialize_h(w, k)
    result = symnmfmodule.symnmf(h.tolist(), w, MAX_ITER, EPSILON)
    return np.argmax(np.asarray(result), axis=1)


def main():
    try:
        # Get args 
        k = int(sys.argv[1])
        points = np.loadtxt(sys.argv[2], delimiter=",", dtype=float, ndmin=2)
        # Calculate labels
        nmf = symnmf_labels(points, k)
        kmeans = kmeans_labels(points, k)
        
        print(f"nmf: {silhouette_score(points, nmf):.4f}")
        print(f"kmeans: {silhouette_score(points, kmeans):.4f}")
        print(f"ari: {adjusted_rand_score(nmf, kmeans):.4f}")
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()