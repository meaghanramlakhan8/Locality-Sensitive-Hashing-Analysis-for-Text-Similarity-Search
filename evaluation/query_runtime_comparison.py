import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing.preprocessing import get_data, preprocess
from lsh_methods.lsh_methods import kmeans_lsh
from evaluation.evaluationsrp import signed_random_projections

def simulate_query_runtime(tfidf_matrix, dataset_sizes, srp_query_func, kmeans_query_func, srp_params, kmeans_params):
    """
    Measure query runtime for SRP-LSH and K-means LSH across different dataset sizes.

    Params:
        - tfidf_matrix: Full TF-IDF matrix for the dataset.
        - dataset_sizes: List of dataset sizes to test (e.g., [500, 1000, 2000]).
        - srp_query_func: Function to perform SRP-LSH querying.
        - kmeans_query_func: Function to perform K-means LSH querying.
        - srp_params: Parameters for SRP-LSH query function.
        - kmeans_params: Parameters for K-means LSH query function.

    Returns:
        - query_times_srp: List of query times for SRP-LSH.
        - query_times_kmeans: List of query times for K-means LSH.
    """
    query_times_srp = []
    query_times_kmeans = []

    for size in dataset_sizes:
        print(f"Testing with dataset size: {size}")
        subset_matrix = tfidf_matrix[:size]  # Subset the dataset
        query_vector = subset_matrix[0]  # Query the first document

        # Measure SRP-LSH query time
        start_time = time.time()
        srp_query_func(subset_matrix, query_vector, **srp_params)
        query_times_srp.append(time.time() - start_time)

        # Measure K-means LSH query time
        start_time = time.time()
        kmeans_query_func(subset_matrix, query_vector, **kmeans_params)
        query_times_kmeans.append(time.time() - start_time)

    return query_times_srp, query_times_kmeans

def plot_query_efficiency(dataset_sizes, query_times_srp, query_times_kmeans):
    """
    Plot query time vs. dataset size for SRP-LSH and K-means LSH.

    Params:
        - dataset_sizes: List of dataset sizes.
        - query_times_srp: Query times for SRP-LSH.
        - query_times_kmeans: Query times for K-means LSH.
    """
    plt.figure(figsize=(10, 6))
    
    # SRP-LSH line
    plt.plot(dataset_sizes, query_times_srp, label="SRP-LSH", color="blue", marker="o", linewidth=2)
    
    # K-means LSH line
    plt.plot(dataset_sizes, query_times_kmeans, label="K-means LSH", color="orange", marker="s", linewidth=2)
    
    plt.title("Query Time vs. Dataset Size", fontsize=16)
    plt.xlabel("Dataset Size", fontsize=14)
    plt.ylabel("Query Time (seconds)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def srp_query_func(matrix, query_vector, n_planes=7):
    """
    Simulate SRP-LSH querying.
    This would involve hashing the query and finding nearest neighbors.
    """
    # Generate SRP hash codes
    hash_codes = signed_random_projections(matrix, n_planes)
    # Find documents in the same hash bucket as the query
    query_hash = signed_random_projections(query_vector.reshape(1, -1), n_planes)
    return hash_codes == query_hash  # Placeholder: Use actual bucket matching logic

def kmeans_query_func(matrix, query_vector, n_clusters=10):
    """
    Simulate K-means LSH querying.
    This would involve assigning the query to a cluster and finding nearest neighbors within the cluster.
    """
    # Cluster assignments for the matrix
    cluster_labels = kmeans_lsh(matrix, n_clusters=n_clusters)
    # Assign the query vector to the nearest cluster
    return cluster_labels[0]  # Placeholder: Refine with distance comparison

if __name__ == "__main__":
    # Load and preprocess data
    print("Fetching and preprocessing data...")
    texts, labels, target_names = get_data()
    tfidf_matrix, vectorizer, categories_of_documents = preprocess(texts, labels, target_names)
    print("Data preprocessing completed.")

    