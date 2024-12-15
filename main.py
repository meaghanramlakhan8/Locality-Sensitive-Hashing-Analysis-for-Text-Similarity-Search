from preprocessing.preprocessing import get_data, preprocess
from evaluation.precision_recall import plot_comparative_precision_recall
from evaluation.evaluationsrp import signed_random_projections, plot_similarity_vs_planes, visualize_srp_with_categories, plot_similarity_to_srp_centroids, plot_retrieval_vs_buckets
from lsh_methods.lsh_methods import kmeans_lsh
from evaluation.evaluation_kmeans import plot_clusters, plot_radial_clusters, plot_by_frequency, write_clusters_to_file, visualize_cluster_counts
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def main():
    # Load and preprocess data
    print("Fetching and preprocessing data...")
    texts, labels, target_names = get_data()
    tfidf_matrix, vectorizer, categories_of_documents = preprocess(texts, labels, target_names)
    print("Data preprocessing completed.")

    ### Section for K-means LSH ###
    print("Applying K-means LSH...")
    kmeans_labels = kmeans_lsh(tfidf_matrix)
    print("K-means LSH applied successfully.")

    # Visualize K-means results
    plot_clusters(tfidf_matrix, kmeans_labels, categories_of_documents)
    plot_radial_clusters(kmeans_labels, categories_of_documents)
    visualize_cluster_counts(kmeans_labels, categories_of_documents)
    write_clusters_to_file(kmeans_labels, categories_of_documents)

    # Additional K-means insights
    print("\n--- K-means Insights ---")
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print(f"Number of Clusters: {len(np.unique(kmeans_labels))}")

    ### Section for SRP-LSH ###
    print("Applying Signed Random Projections LSH...")
    srp_hashes = signed_random_projections(tfidf_matrix, n_planes=7)
    print(f"SRP Hash Codes (first 5 documents): {srp_hashes[:5]}")

    # Visualize SRP results
    visualize_srp_with_categories(tfidf_matrix, srp_hashes, labels, target_names, include_centroids=True)
    plot_similarity_to_srp_centroids(tfidf_matrix, srp_hashes)

    # SRP performance with varying hyperplanes
    print("Analyzing SRP performance with varying hyperplanes...")
    n_planes_range = range(2, 21, 2)
    plot_similarity_vs_planes(tfidf_matrix, n_planes_range)
    plot_retrieval_vs_buckets(tfidf_matrix, labels, n_planes_range)

    ### Precision-Recall Comparison ###
    print("Performing precision-recall comparison...")
    # Generate similarity scores for precision-recall comparison
    y_true = np.array([1 if labels[i] == labels[j] else 0 for i in range(len(labels)) for j in range(len(labels))])
    scores_srp = cosine_similarity(tfidf_matrix).flatten()  # Placeholder for SRP similarities
    scores_kmeans = cosine_similarity(tfidf_matrix).flatten()  # Placeholder for K-means similarities

    plot_comparative_precision_recall(y_true, scores_srp, scores_kmeans)

    ### Overall Data Visualizations ###
    print("Plotting top terms by frequency...")
    plot_by_frequency(tfidf_matrix, vectorizer)

if __name__ == "__main__":
    main()
