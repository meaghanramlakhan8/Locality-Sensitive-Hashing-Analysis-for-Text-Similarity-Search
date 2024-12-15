import time
import numpy as np
import matplotlib.pyplot as plt
from lsh_methods.lsh_methods import signed_random_projections_lsh
from sklearn.metrics.pairwise import cosine_similarity
from lsh_methods.lsh_methods import kmeans_lsh


def simulate_query_runtime(tfidf_matrix, dataset_sizes, srp_query_func, kmeans_query_func, srp_params, kmeans_params):
    """
    Measure query runtime for SRP-LSH and K-means LSH across different dataset sizes.

    Params:
        - tfidf_matrix: Full preprocessed TF-IDF matrix for the dataset.
        - dataset_sizes: List of dataset sizes to test.
        - srp_query_func: Function to perform SRP-LSH querying.
        - kmeans_query_func: Function to perform K-means LSH querying.
        - srp_params: Parameters for SRP-LSH query function.
        - kmeans_params: Parameters for K-means LSH query function.

    Returns:
        - query_times_srp: List of query times for SRP across dataset sizes.
        - query_times_kmeans: List of query times for K-means across dataset sizes.
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

def srp_query_func(matrix, query_vector, n_planes=10):
    """
    Perform a full SRP-LSH query.
    """
    # Generate SRP hash codes
    hash_codes = signed_random_projections_lsh(matrix, n_planes)
    query_hash = signed_random_projections_lsh(query_vector.reshape(1, -1), n_planes)

    # Find documents in the same bucket
    bucket_indices = np.where((hash_codes == query_hash).all(axis=1))[0]
    bucket_docs = matrix[bucket_indices]

    # Compute similarity of the query to documents in the bucket
    if bucket_docs.shape[0] > 0:  # Use shape[0] to check the number of rows/documents
        similarities = cosine_similarity(bucket_docs, query_vector.reshape(1, -1))
    return bucket_indices  # Return bucket indices (or other desired outputs)


def kmeans_query_func(matrix, query_vector, n_clusters=7):
    """
    Simulate K-means LSH querying.
    This would involve assigning the query to a cluster and finding nearest neighbors within the cluster.
    """
    # Cluster assignments for the matrix
    cluster_labels = kmeans_lsh(matrix, n_clusters=n_clusters)

    # Find the cluster of the query vector
    query_cluster_label = cluster_labels[0]
    cluster_docs_indices = np.where(cluster_labels == query_cluster_label)[0]
    cluster_docs = matrix[cluster_docs_indices]

    # Compute similarity of the query to documents in the cluster
    if cluster_docs.shape[0] > 0:  # Use shape[0] to check the number of rows/documents
        similarities = cosine_similarity(cluster_docs, query_vector.reshape(1, -1))
    return cluster_docs_indices  # Return cluster indices (or other desired outputs)


def plot_query_time_real(results):
    """
    Plot real query times for SRP and K-means.

    Params:
        - results: Dictionary containing query times for different dataset sizes.
    """
    dataset_sizes = results["dataset_sizes"]
    
    plt.figure(figsize=(12, 8))
    
    # Real query times
    plt.plot(dataset_sizes, results["real_srp"], label="Real SRP Query Time", color="blue", marker="o", linewidth=2)
    plt.plot(dataset_sizes, results["real_kmeans"], label="Real K-means Query Time", color="orange", marker="s", linewidth=2)
    
    # Formatting
    plt.title("Query Time: Real SRP vs. Real K-means", fontsize=16)
    plt.xlabel("Dataset Size", fontsize=14)
    plt.ylabel("Query Time (seconds)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    #Setting output directory to be comparison_plots
    comparison_plots_dir = os.path.join(os.getcwd(), "plots/comparison_plots") 
    plt.savefig(os.path.join(comparison_plots_dir, "compute_lsh_precisions.png"))
    plt.show()