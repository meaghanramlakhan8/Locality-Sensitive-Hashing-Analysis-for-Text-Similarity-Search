import numpy as np
import os
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def signed_random_projections(tfidf_matrix, n_planes=7):
    """
    Perform Signed Random Projections (SRP).

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - n_planes: Number of random hyperplanes.

    Returns:
        - hash_codes: Binary hash codes for each document.
    """
    random_planes = np.random.randn(n_planes, tfidf_matrix.shape[1])  # Generate random hyperplanes
    projections = tfidf_matrix.dot(random_planes.T)  # Project data onto hyperplanes
    hash_codes = (projections > 0).astype(int)  # Convert projections to binary hash codes
    return hash_codes

def compute_srp_centroids(tfidf_matrix, hash_codes):
    """
    Compute centroids for each SRP hash bucket.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - hash_codes: Binary hash codes for each document.

    Returns:
        - centroids: A dictionary mapping hash bucket IDs to their centroids.
        - bucket_assignments: A dictionary mapping hash bucket IDs to document indices.
    """
    # Convert binary hash codes to integer bucket IDs
    bucket_ids = np.dot(hash_codes, 1 << np.arange(hash_codes.shape[1]))

    # Group documents by their hash bucket
    unique_buckets = np.unique(bucket_ids)
    bucket_assignments = {bucket: np.where(bucket_ids == bucket)[0] for bucket in unique_buckets}

    # Compute centroids for each bucket
    centroids = {}
    for bucket, indices in bucket_assignments.items():
        bucket_matrix = tfidf_matrix[indices]
        centroids[bucket] = bucket_matrix.mean(axis=0)

    return centroids, bucket_assignments

def write_srp_clusters_to_file(tfidf_matrix, hash_codes, categories_of_documents, output_file="srp_results.txt"):
    """
    Write SRP hash buckets and their corresponding documents along with categories to a file.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - hash_codes: Binary hash codes for each document.
        - categories_of_documents: Mapping of the categories to the list of indices (documents).
        - output_file: Name of the output file to write to.
    """
    # Convert hash codes to integer bucket IDs
    bucket_ids = np.dot(hash_codes, 1 << np.arange(hash_codes.shape[1]))

    # Map documents to their categories
    doc_to_category = {}
    for category, doc_indices in categories_of_documents.items():
        for doc_idx in doc_indices:
            doc_to_category[doc_idx] = category

    # Group documents by their hash buckets
    unique_buckets = np.unique(bucket_ids)
    bucket_assignments = {bucket: np.where(bucket_ids == bucket)[0] for bucket in unique_buckets}

    # Write the results to the file
    with open(output_file, "w") as f:
        for bucket, indices in bucket_assignments.items():
            f.write(f"Hash Bucket {bucket}:\n")
            category_counts = {}

            for doc_idx in indices:
                category = doc_to_category.get(doc_idx, "Unknown")
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1

            # Write category counts
            f.write("  Category Counts:\n")
            for category, count in category_counts.items():
                f.write(f"    {category}: {count}\n")

            # Write document details
            f.write("  Documents:\n")
            for doc_idx in indices:
                category = doc_to_category.get(doc_idx, "Unknown")
                f.write(f"    Document {doc_idx} - Category: {category}\n")
            f.write("\n")
    
    print(f"SRP cluster details written to {output_file}")

def plot_similarity_to_srp_centroids(tfidf_matrix, hash_codes):
    """
    Plot the similarity of each document to the centroid of its SRP hash bucket,
    including a histogram, KDE curve, and mean/median lines, using Matplotlib and Scipy.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - hash_codes: Binary hash codes for each document.
    """
    centroids, bucket_assignments = compute_srp_centroids(tfidf_matrix, hash_codes)

    # Compute similarities of documents to their assigned centroid
    similarities = []
    for bucket, indices in bucket_assignments.items():
        bucket_matrix = tfidf_matrix[indices]
        centroid = centroids[bucket]

        bucket_matrix_dense = bucket_matrix.toarray()
        centroid_dense = np.asarray(centroid).reshape(1, -1)
        similarities.extend(cosine_similarity(bucket_matrix_dense, centroid_dense).flatten())

    # Calculate mean and median
    mean_similarity = np.mean(similarities)
    median_similarity = np.median(similarities)

    # Create histogram
    plt.figure(figsize=(12, 8))
    n, bins, _ = plt.hist(similarities, bins=30, alpha=0.7, color="blue", edgecolor="black", label="Similarity Distribution")

    kde = gaussian_kde(similarities)
    x_kde = np.linspace(min(similarities), max(similarities), 500)
    y_kde = kde(x_kde)

    # Plot KDE curve
    plt.plot(x_kde, y_kde * len(similarities) * (bins[1] - bins[0]), color="orange", label="KDE Curve", linewidth=2)

    plt.axvline(mean_similarity, color="red", linestyle="--", label=f"Mean: {mean_similarity:.2f}")
    plt.axvline(median_similarity, color="green", linestyle="--", label=f"Median: {median_similarity:.2f}")

    plt.title("Document Similarity to SRP Hash Bucket Centroid", fontsize=16)
    plt.xlabel("Cosine Similarity", fontsize=14)
    plt.ylabel("Number of Documents", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    SRP_plots_dir = os.path.join(os.getcwd(), "plots/SRP_plots")
    os.makedirs(SRP_plots_dir, exist_ok=True)
    plt.savefig(os.path.join(SRP_plots_dir, "plot_similarity_to_srp_centroids.png"))
    plt.show()

if __name__ == "__main__":
    # Example usage
    # Assuming tfidf_matrix and categories_of_documents are already defined
    tfidf_matrix = ...  # Replace with your actual sparse matrix
    categories_of_documents = ...  # Replace with your actual category mapping

    # Apply SRP
    hash_codes = signed_random_projections(tfidf_matrix, n_planes=7)

    # Write SRP clusters to file
    write_srp_clusters_to_file(tfidf_matrix, hash_codes, categories_of_documents)

    # Optional: Plot similarity to centroids
    plot_similarity_to_srp_centroids(tfidf_matrix, hash_codes)
