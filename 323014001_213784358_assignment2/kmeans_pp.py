import sys
import numpy as np
import pandas as pd
# import mykmeanssp

def euclidean_distance(p, q):
    sum_sq = 0.0
    for i in range(len(p)):
        sum_sq += (p[i] - q[i]) ** 2
    return sum_sq ** 0.5

def read_points_file(path):
    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)
    return df

def inner_join_and_sort(df1, df2): 
    merged = pd.merge(df1, df2, on=0, how="inner")
    merged = merged.sort_values(by=0, ascending=True).reset_index(drop=True)
 
    keys = merged[0].tolist()
    feature_cols = [c for c in merged.columns if c != 0]
    datapoints = merged[feature_cols].values.tolist()
 
    return keys, datapoints
    
def parse_args():
    argv = sys.argv[1:]

    if len(argv) == 5:
        k_arg, iter_arg, eps_arg, file1, file2 = argv
    elif len(argv) == 4:
        k_arg = "3" # default if not given
        iter_arg, eps_arg, file1, file2 = argv
    else:
        print("An Error Has Occurred")
        sys.exit(1)

    if not k_arg.isdigit(): # to check if is natural number - <N will be checked later
        print("Incorrect number of clusters!")
        sys.exit(1)
    k = int(k_arg)

    if not iter_arg.isdigit(): # to check if is natural number
        print("Incorrect maximum iteration!")
        sys.exit(1)
    iterations = int(iter_arg)
    if not (1 < iterations < 400): # to check if is 1<.<400
        print("Incorrect maximum iteration!")
        sys.exit(1)

    try:
        eps = float(eps_arg)
    except ValueError:
        print("Incorrect epsilon!")
        sys.exit(1)
    if eps < 0: # to check if is >= 0
        print("Incorrect epsilon!")
        sys.exit(1)

    return k, iterations, eps, file1, file2

def main():
    # Get and validate args
    k, iterations, eps, file1, file2 = parse_args() 

    # Read and join data
    data1 = read_points_file(file1)
    data2 = read_points_file(file2)
    keys, datapoints = inner_join_and_sort(data1, data2)

    N = len(datapoints)
    if N == 0:
        print("An Error Has Occurred")
        return

    if not (1 < k < N): # last input check
        print("Incorrect number of clusters!")
        return

    # Initialize the centroids
    np.random.seed(1234)  # as told
    points = np.array(datapoints, dtype=float)
    N = points.shape[0]

    chosen_indices = []

    first_idx = int(np.random.choice(N)) 
    chosen_indices.append(first_idx)

    D = np.full(N, np.inf) # D(x) initialization to inf

    for _ in range(1, k):
        last_centroid = points[chosen_indices[-1]]
        dist_to_last = np.linalg.norm(points - last_centroid, axis=1)
        D = np.minimum(D, dist_to_last) # D(x) calc

        probs = D / D.sum()
        next_idx = int(np.random.choice(N, p=probs))
        chosen_indices.append(next_idx)

    initial_centroids = [datapoints[i] for i in chosen_indices]

    # Calling the c module for calc
    final_centroids = mykmeanssp.fit(
        k,
        iterations,
        eps,
        initial_centroids,
        datapoints,
    )

    # Finish
    print(",".join(str(int(keys[i])) for i in chosen_indices))
    for centroid in final_centroids:
        print(",".join(f"{val:.4f}" for val in centroid))


if __name__ == "__main__":
    main()