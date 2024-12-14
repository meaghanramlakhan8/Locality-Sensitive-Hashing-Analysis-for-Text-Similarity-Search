from preprocessing.preprocessing import get_data, preprocess
import matplotlib.pyplot as plt
from evaluation.evaluationsrp import signed_random_projections, visualize_srp_with_categories, compute_srp_centroids
from lsh_methods.lsh_methods import kmeans_lsh
from evaluation.evaluation import evaluate_retrieval, plot_clusters, plot_radial_clusters, plot_by_frequency, plot_silhouette, write_clusters_to_file
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def plot_similarity_vs_planes(tfidf_matrix, n_planes_range):
    """
    Plot the mean cosine similarity vs. number of hyperplanes for SRP.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - n_planes_range: List or range of hyperplane counts to test.
    """
    mean_similarities = []

    for n_planes in n_planes_range:
        print(f"Testing SRP with {n_planes} hyperplanes...")
        # Perform SRP with current number of hyperplanes
        hash_codes = signed_random_projections(tfidf_matrix, n_planes)
        
        # Compute SRP centroids and similarities
        centroids, bucket_assignments = compute_srp_centroids(tfidf_matrix, hash_codes)
        similarities = []
        for bucket, indices in bucket_assignments.items():
            bucket_matrix = tfidf_matrix[indices]
            centroid = centroids[bucket]

            # Convert data to dense format
            bucket_matrix_dense = bucket_matrix.toarray()
            centroid_dense = np.asarray(centroid).reshape(1, -1)

            # Compute cosine similarities
            similarities.extend(cosine_similarity(bucket_matrix_dense, centroid_dense).flatten())
        
        # Store the mean similarity for this n_planes
        mean_similarities.append(np.mean(similarities))
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(n_planes_range, mean_similarities, marker='o', color='blue', label="Mean Similarity")
    plt.title("Mean Cosine Similarity vs. Number of SRP Hyperplanes", fontsize=16)
    plt.xlabel("Number of Hyperplanes (n_planes)", fontsize=14)
    plt.ylabel("Mean Cosine Similarity", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.show()

def main():
    # Load and preprocess data for K-means
    print("Fetching and preprocessing data for K-means LSH...")
    texts, labels, target_names = get_data(sample_size=5000)
    tfidf_matrix, vectorizer, categories_of_documents = preprocess(texts, labels, target_names)
    print("Data preprocessing completed.")

    # Section for K-means LSH
    print("Applying K-means LSH...")
    kmeans_labels = kmeans_lsh(tfidf_matrix)
    print(f"K-means Labels (first 10):\n{kmeans_labels[:10]}")
    plot_clusters(tfidf_matrix, kmeans_labels, categories_of_documents)
    plot_radial_clusters(kmeans_labels, categories_of_documents)
    plot_silhouette(tfidf_matrix, kmeans_labels, categories_of_documents, target_names)

    # Evaluate retrieval metrics for K-means LSH
    print("Evaluating K-means LSH...")
    kmeans_metrics = evaluate_retrieval(labels, kmeans_labels)
    print(f"K-means LSH Metrics:\n{kmeans_metrics}")

    # Compute and print cosine similarity
    print("Calculating cosine similarity matrix...")
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"Cosine Similarity Matrix (first 5 rows):\n{similarity_matrix[:5, :5]}")

    # Section for SRP-LSH
    print("Applying Signed Random Projections LSH...")
    srp_hashes = signed_random_projections(tfidf_matrix, n_planes=7)
    print(f"SRP Hash Codes (first 5 documents):\n{srp_hashes[:5]}")
    
    # Visualize SRP results
    print("Visualizing SRP-LSH results...")
    visualize_srp_with_categories(tfidf_matrix, srp_hashes, labels, target_names)

    # Plot similarity vs. number of hyperplanes
    print("Analyzing SRP performance with varying hyperplanes...")
    n_planes_range = range(2, 21, 2)  # Test with hyperplanes from 2 to 20, stepping by 2
    plot_similarity_vs_planes(tfidf_matrix, n_planes_range)

    # Plot term frequency for the dataset
    print("Plotting term frequencies...")
    plot_by_frequency(tfidf_matrix, vectorizer)

    # Write clusters to file
    print("Writing clusters to file...")
    write_clusters_to_file(kmeans_labels, categories_of_documents)

    # Additional K-means visualization outputs
    print("\n--- Additional Information ---")
    print("Categories of Documents:\n", categories_of_documents)
    print("\nTF-IDF Matrix Shape:\n", tfidf_matrix.shape)
    print("\nK-means Labels:\n", kmeans_labels)
    print("Number of Clusters:", len(np.unique(kmeans_labels)))

if __name__ == "__main__":
    main()
