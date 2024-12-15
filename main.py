from preprocessing.preprocessing import get_data, preprocess
import matplotlib.pyplot as plt
from evaluation.evaluationsrp import signed_random_projections, plot_similarity_vs_planes, visualize_srp_with_categories, plot_similarity_to_srp_centroids, plot_retrieval_vs_buckets
from lsh_methods.lsh_methods import kmeans_lsh
from evaluation.evaluation import plot_clusters, plot_radial_clusters, plot_by_frequency, write_clusters_to_file, visualize_cluster_counts
import numpy as np

def main():
    # Load and preprocess data !!!!
    texts, labels, target_names = get_data()
    tfidf_matrix, vectorizer, categories_of_documents = preprocess(texts, labels, target_names)
    print("Data preprocessing completed.")
    
    ### Beginning of section for K-means LSH
    kmeans_labels = kmeans_lsh(tfidf_matrix)
    print("Applied K-means LSH.")
    #plot_clusters(tfidf_matrix, kmeans_labels, categories_of_documents)  # Cluster plot for K-means
    #plot_radial_clusters(kmeans_labels, categories_of_documents)  # Radial clusters plot for K-means
    #write_clusters_to_file(kmeans_labels, categories_of_documents)  # Outputs counts of categories per cluster into file
    #visualize_cluster_counts(kmeans_labels, categories_of_documents)  # Visualization of counts of categories per cluster

    # Additional print statements for K-means visualization of data
    print("\n--- Additional Information ---")
    print("Categories of Documents:\n", categories_of_documents)
    print("\nTF-IDF Matrix Shape:\n", tfidf_matrix.shape)
    print("\nK-means Labels:\n", kmeans_labels)
    print("Number of Clusters:", len(np.unique(kmeans_labels)))
    ### End of section for K-means LSH

    ### Beginning of section for SRP-LSH
    print("Applying Signed Random Projections LSH...")
    srp_hashes = signed_random_projections(tfidf_matrix, n_planes=7)
    print(f"SRP Hash Codes (first 5 documents):\n{srp_hashes[:5]}")

    # Visualize SRP results
    visualize_srp_with_categories(tfidf_matrix, srp_hashes, labels, target_names, include_centroids=True)

    # Plot similarity to SRP centroids with KDE
    plot_similarity_to_srp_centroids(tfidf_matrix, srp_hashes)

    # Analyze SRP performance with varying hyperplanes
    print("Analyzing SRP performance with varying hyperplanes...")
    n_planes_range = range(2, 21, 2)  # Test with hyperplanes from 2 to 20, stepping by 2
    plot_similarity_vs_planes(tfidf_matrix, n_planes_range)  # Plot similarity vs. number of hyperplanes

    # Plot retrieval precision vs. number of SRP hyperplanes
    print("Plotting retrieval precision vs. SRP hyperplanes...")
    plot_retrieval_vs_buckets(tfidf_matrix, labels, n_planes_range)
    ### End of section for SRP-LSH

    # Section for overall data visualizations
    plot_by_frequency(tfidf_matrix, vectorizer)  # Outputs the top 25 words across all data

if __name__ == "__main__":
    main()
