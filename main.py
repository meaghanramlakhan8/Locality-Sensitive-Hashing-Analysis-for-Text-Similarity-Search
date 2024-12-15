from lsh_methods.lsh_methods import kmeans_lsh, signed_random_projections_lsh
from preprocessing.preprocessing import get_data, preprocess
from evaluation.evaluationsrp import visualize_srp_with_categories, plot_similarity_to_srp_centroids, plot_similarity_vs_planes, write_srp_clusters_to_file
from evaluation.evaluation_kmeans import plot_clusters, plot_radial_clusters, plot_by_frequency, write_clusters_to_file, visualize_cluster_counts
from evaluation.precision_recall import generate_ground_truth, compute_similarity_scores, precision_recall_curve
from evaluation.evaluation_comparison import compute_lsh_precisions
from evaluation.query_runtime_comparison import simulate_query_runtime, srp_query_func, kmeans_query_func, plot_query_time_real
from evaluation.memory_usage import measure_memory_usage, plot_memory_usage
from evaluation.scatterplot import plot_3d_comparative_metrics
from sklearn.metrics import precision_recall_curve, auc
import numpy as np


def main():
    # Load and preprocess data
    texts, labels, target_names = get_data()
    tfidf_matrix, vectorizer, categories_of_documents = preprocess(texts, labels, target_names)
    print("Data preprocessing completed.")

    ### Begining of section for K-means LSH ###
    kmeans_labels = kmeans_lsh(tfidf_matrix)
    print("Applied K-means LSH.")
    plot_clusters(tfidf_matrix, kmeans_labels, categories_of_documents)  # Cluster plot for K-means
    plot_radial_clusters(kmeans_labels, categories_of_documents)  # Radial clusters plot for K-means
    write_clusters_to_file(kmeans_labels, categories_of_documents)  # Outputs counts of categories per cluster into file
    visualize_cluster_counts(kmeans_labels, categories_of_documents)  # Visualization of counts of categories per cluster

    # Debugging output for K-means
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print(f"Number of Clusters: {len(np.unique(kmeans_labels))}")
    ### End of section for K-means LSH ###

    ### Beginning of section for SRP-LSH ###
    print("Applying Signed Random Projections LSH...")
    srp_hashes = signed_random_projections_lsh(tfidf_matrix, n_planes=7)
    write_srp_clusters_to_file(tfidf_matrix, srp_hashes, categories_of_documents)
    visualize_srp_with_categories(tfidf_matrix, srp_hashes, labels, target_names)  # Visualize SRP results
    plot_similarity_to_srp_centroids(tfidf_matrix, srp_hashes)  # Plot similarity to SRP centroids with KDE

    n_planes_range = range(2, 21, 2)  # Test with hyperplanes from 2 to 20, stepping by 2
    plot_similarity_vs_planes(tfidf_matrix, n_planes_range)  # Analyze SRP performance with varying hyperplanes
    ### End of section for SRP-LSH ###

    ### Beginning of section for overall data visualizations ###
    plot_by_frequency(tfidf_matrix, vectorizer)  # Outputs the top 25 words across all data
    compute_lsh_precisions(tfidf_matrix, categories_of_documents)  # Gets the precision of both LSHs based on different clusters/planes
    ### End of section for overall data visualizations ###

    ### Define dataset sizes and query parameters ###
    dataset_sizes = [2500, 5000, 7500, 10000, 12500, 15000, 18000]
    srp_params = {"n_planes": 10}
    kmeans_params = {"n_clusters": 7}

    # Measure query times
    query_times_srp, query_times_kmeans = simulate_query_runtime(
        tfidf_matrix,
        dataset_sizes,
        srp_query_func=srp_query_func,
        kmeans_query_func=kmeans_query_func,
        srp_params=srp_params,
        kmeans_params=kmeans_params,
    )

    # Create results dictionary for plotting
    results = {
        "dataset_sizes": dataset_sizes,
        "real_srp": query_times_srp,
        "real_kmeans": query_times_kmeans,
    }

    # Plot query times
    plot_query_time_real(results)

    # Measure memory usage for K-means LSH
    memory_usage_kmeans = measure_memory_usage(kmeans_lsh, tfidf_matrix, n_clusters=7)

    # Measure memory usage for SRP-LSH
    memory_usage_srp = measure_memory_usage(signed_random_projections_lsh, tfidf_matrix, n_planes=10)

    # Plot memory usage comparison
    plot_memory_usage(memory_usage_kmeans, memory_usage_srp)


if __name__ == "__main__":
    main()
