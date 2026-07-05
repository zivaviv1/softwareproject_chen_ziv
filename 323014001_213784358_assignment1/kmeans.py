import sys

def euclidean_distance(p, q):
    sum_sq = 0.0
    for i in range(len(p)):
        sum_sq += (p[i] - q[i]) ** 2
    return sum_sq ** 0.5

def main():
    # Validate inputs
    if len(sys.argv) < 2:
        print("Please provide a valid integer for k.")
        return
    if len(sys.argv) > 3:
        print("An Error Has Occurred")
        return

    k_arg = sys.argv[1]
    if not k_arg.isdigit():
        print("Incorrect number of clusters!")
        return
    k = int(k_arg)

    if len(sys.argv) == 3:
        iter_arg = sys.argv[2]
        if not iter_arg.isdigit():
            print("Incorrect maximum iteration!")
            return
        iter = int(iter_arg)
    else:
        iter = 400

    # Read data
    lines = sys.stdin.readlines()
    lines = [line.strip() for line in lines]
    datapoints = [[float(x) for x in line.split(",")] for line in lines]
    N = len(datapoints)
    
    if k <= 1 or k >= N:
        print("Incorrect number of clusters!")
        return
    if iter >= 800 or iter <= 1:
        print("Incorrect maximum iteration!")
        return
    
    if not datapoints:
        return
        
    d = len(datapoints[0]) # Number of dimensions (e.g., 3)
    epsilon = 0.001
    
    # Initialize the centroids
    centroids = [datapoints[i] for i in range(k)]
    
    # Run the algorithm 
    for iteration in range(iter):
        clusters = [[] for _ in range(k)]
        
        # Assign every datapoint to the closest cluster
        for x in datapoints:
            min_dist = float('inf')
            closest_k = 0
            
            for i in range(k):
                dist = euclidean_distance(x, centroids[i])
                if dist < min_dist:
                    min_dist = dist
                    closest_k = i
                    
            clusters[closest_k].append(x)
            
        # Update the centroids
        new_centroids = []
        max_delta = 0.0
        
        for i in range(k):
            cluster = clusters[i]
            
            if len(cluster) == 0:
                new_centroids.append(centroids[i])
                continue
                
            new_centroid = [0.0] * d
            for x in cluster:
                for dim in range(d):
                    new_centroid[dim] += x[dim]
            
            for dim in range(d):
                new_centroid[dim] /= len(cluster)
                
            new_centroids.append(new_centroid)
            
            # Check convergence delta for this centroid
            delta = euclidean_distance(new_centroid, centroids[i])
            if delta > max_delta:
                max_delta = delta
                
        centroids = new_centroids
        
        # Check for convergence
        if max_delta < epsilon:
            break
            
    # Finish
    for centroid in centroids:
        print(",".join(f"{val:.4f}" for val in centroid))

if __name__ == "__main__":
    main()
    
