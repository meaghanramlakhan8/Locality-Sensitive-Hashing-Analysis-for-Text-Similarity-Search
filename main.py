from preprocessing.preprocessing import get_data, preprocess
import matplotlib.pyplot as plt
from evaluation.evaluationsrp import signed_random_projections, visualize_srp_with_categories, compute_srp_centroids, plot_similarity_vs_planes
from lsh_methods.lsh_methods import kmeans_lsh
from evaluation.evaluation import evaluate_retrieval, plot_clusters, plot_radial_clusters, plot_by_frequency, write_clusters_to_file
from evaluation.evaluation import visualize_cluster_counts
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

def main():
    # Load and preprocess data !!!!
    texts, labels, target_names = get_data()
    tfidf_matrix, vectorizer, categories_of_documents = preprocess(texts, labels, target_names)
    print("Data preprocessing completed.")
    

    ### Beginning of section for K-means LSH
    kmeans_labels = kmeans_lsh(tfidf_matrix)
    print("Applied K-means LSH.")
    plot_clusters(tfidf_matrix, kmeans_labels, categories_of_documents) #cluster plot for k-means
    plot_radial_clusters(kmeans_labels, categories_of_documents)    #radial clusters plot for k-means
    write_clusters_to_file(kmeans_labels, categories_of_documents) #outputs counts of categories per cluster into file
    visualize_cluster_counts(kmeans_labels, categories_of_documents) #visualization of the counts of categories per cluster 

    #(Additional print statements for K-means visualization of data)
    print("\n--- Additional Information ---")
    print("Categories of Documents:\n", categories_of_documents)
    print("\nTF-IDF Matrix Shape:\n", tfidf_matrix.shape)
    print("\nK-means Labels:\n", kmeans_labels)
    print("Number of Clusters:", len(np.unique(kmeans_labels)))
    ### End of section for K-means LSH


    ### Beginning of section for SRP-LSH
    similarity_matrix = cosine_similarity(tfidf_matrix)  # Compute and print cosine similarity
    print("Applying Signed Random Projections LSH...")
    srp_hashes = signed_random_projections(tfidf_matrix, n_planes=7)
    print(f"SRP Hash Codes (first 5 documents):\n{srp_hashes[:5]}")
    visualize_srp_with_categories(tfidf_matrix, srp_hashes, labels, target_names) # Visualize SRP results
    print("Analyzing SRP performance with varying hyperplanes...")
    n_planes_range = range(2, 21, 2)  # Test with hyperplanes from 2 to 20, stepping by 2
    plot_similarity_vs_planes(tfidf_matrix, n_planes_range) # Plot similarity vs. number of hyperplanes
    ### End of section for SRP-LSH



    # Section for overall data visualizations
    plot_by_frequency(tfidf_matrix, vectorizer) #outputs the top 25 words across all data

if __name__ == "__main__":
    main()
