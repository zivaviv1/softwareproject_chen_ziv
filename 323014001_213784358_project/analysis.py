import sys
import io
import contextlib
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
import symnmfmodule
from kmeans import calculate_kmeans

np.random.seed(1234)

EPSILON = 1e-4
MAX_ITER = 300

def _initialize_h(w, k):
    m = float(np.mean(w))
    return np.random.uniform(0, 2 * np.sqrt(m / k), size=(len(w), k))


def symnmf_labels(points, k):
    w = symnmfmodule.norm(points.tolist())
    h = _initialize_h(w, k)
    result = symnmfmodule.symnmf(h.tolist(), w, MAX_ITER, EPSILON)
    return np.argmax(np.asarray(result), axis=1)


def kmeans_labels(points, k):
    # Use the kmeans from HW1
    with contextlib.redirect_stdout(io.StringIO()):
        centroids = calculate_kmeans(k, MAX_ITER, points.tolist())
    if centroids is None:
        raise ValueError()
    distances = np.sum((points[:, None] - np.array(centroids)) ** 2, axis=2)
    return np.argmin(distances, axis=1)


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