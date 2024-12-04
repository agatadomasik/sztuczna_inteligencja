import numpy as np
import random

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    centroids_indices = random.sample(range(len(data)), k)
    centroids = data[centroids_indices]
    return centroids

def initialize_centroids_kmeans_pp(data, k):
    # TODO implement Unsupervised classification kmeans++ initizalization
    centroids = np.array(data[np.random.choice(data.shape[0], size=1)])
    for _ in range(1, k):
        max_dis = -np.inf
        next_centroid = None
        for observation in data:
            dis = 0
            for centroid in centroids:
                dis += np.sqrt(np.sum((observation-centroid)**2))
            if dis > max_dis:
                max_dis = dis
                next_centroid = observation

        centroids = np.append(centroids, [next_centroid], axis=0)

    return centroids

def assign_to_cluster(data, centroids):
    # TODO find the closest cluster for each data point
    assignments = []
    for observation in data:
        min_dis = np.inf
        centroid_index = 0
        for i, centroid in enumerate(centroids):
            dis = np.sqrt(np.sum((observation-centroid)**2))
            if dis < min_dis:
                min_dis = dis
                centroid_index = i

        assignments.append(centroid_index)

    return assignments

def update_centroids(data, assignments):
    groups = [[] for _ in range(max(assignments) + 1)]
    for i, assignment in enumerate(assignments):
        groups[assignment].append(data[i])

    centroids = []
    for group in groups:
        if group:
            centroid = np.mean(group, axis=0)
            centroids.append(centroid)

    return np.array(centroids)

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus=False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)


    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return assignments, centroids, mean_intra_distance(data, assignments, centroids)