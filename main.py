from lsh_methods.lsh_methods import kmeans_lsh
from preprocessing.preprocessing import get_data, preprocess
from evaluation.evaluationsrp import signed_random_projections, plot_similarity_vs_planes, visualize_srp_with_categories, plot_similarity_to_srp_centroids, plot_retrieval_vs_buckets
from evaluation.evaluation_kmeans import plot_clusters, plot_radial_clusters, plot_by_frequency, write_clusters_to_file, visualize_cluster_counts
from evaluation.evaluation_comparison import plot_comparative_precision_recall
from sklearn.metrics.pairwise import cosine_similarity
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

    # Printing stuff to visualize kmeans 
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print(f"Number of Clusters: {len(np.unique(kmeans_labels))}")
    ### End of section for K-means LSH ###

    ### Section for SRP-LSH ###
    print("Applying Signed Random Projections LSH...")
    srp_hashes = signed_random_projections(tfidf_matrix, n_planes=7)
    visualize_srp_with_categories(tfidf_matrix, srp_hashes, labels, target_names, include_centroids=True) # Visualize SRP results
    plot_similarity_to_srp_centroids(tfidf_matrix, srp_hashes) # Plot similarity to SRP centroids with KDE

    n_planes_range = range(2, 21, 2)  # Test with hyperplanes from 2 to 20, stepping by 2
    plot_similarity_vs_planes(tfidf_matrix, n_planes_range) # Analyze SRP performance with varying hyperplanes

    plot_retrieval_vs_buckets(tfidf_matrix, labels, n_planes_range) # Plot retrieval precision vs. number of SRP hyperplanes
    ### End of section for SRP-LSH ###

    ### Begining of section for overall data visualizations ###
    plot_by_frequency(tfidf_matrix, vectorizer)  # Outputs the top 25 words across all data

    # Generate similarity scores for precision-recall comparison
    y_true = np.array([1 if labels[i] == labels[j] else 0 for i in range(len(labels)) for j in range(len(labels))])
    scores_srp = cosine_similarity(tfidf_matrix).flatten()  # Placeholder for SRP similarities
    scores_kmeans = cosine_similarity(tfidf_matrix).flatten()  # Placeholder for K-means similarities

    
    
    ### End of section for overall data visualizations ###
    plot_comparative_precision_recall(y_true, scores_srp, scores_kmeans)
    # Dataset sizes to test
    dataset_sizes = [500, 1000, 2000, 3000, 4000, 5000]

    # Measure query runtimes
    query_times_srp, query_times_kmeans = simulate_query_runtime(
        tfidf_matrix, dataset_sizes,
        srp_query_func, kmeans_query_func,
        srp_params={'n_planes': 7}, kmeans_params={'n_clusters': 10}
    )

    # Plot query efficiency
    plot_query_efficiency(dataset_sizes, query_times_srp, query_times_kmeans)



if __name__ == "__main__":
    main()
