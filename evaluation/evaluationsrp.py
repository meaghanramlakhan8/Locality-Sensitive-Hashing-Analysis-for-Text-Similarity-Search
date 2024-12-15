import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

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


def plot_similarity_to_srp_centroids(tfidf_matrix, hash_codes):
    """
    Plot the similarity of each document to the centroid of its SRP hash bucket,
    including a histogram, KDE curve, and mean/median lines, using Matplotlib and Scipy.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - hash_codes: Binary hash codes for each document.
    """
    # Compute SRP centroids and bucket assignments
    centroids, bucket_assignments = compute_srp_centroids(tfidf_matrix, hash_codes)

    # Compute similarities of documents to their assigned centroid
    similarities = []
    for bucket, indices in bucket_assignments.items():
        bucket_matrix = tfidf_matrix[indices]
        centroid = centroids[bucket]

        # Convert data to dense format
        bucket_matrix_dense = bucket_matrix.toarray()
        centroid_dense = np.asarray(centroid).reshape(1, -1)

        similarities.extend(cosine_similarity(bucket_matrix_dense, centroid_dense).flatten())

    # Calculate mean and median
    mean_similarity = np.mean(similarities)
    median_similarity = np.median(similarities)

    # Create histogram
    plt.figure(figsize=(12, 8))
    n, bins, _ = plt.hist(similarities, bins=30, alpha=0.7, color="blue", edgecolor="black", label="Similarity Distribution")

    # Compute KDE with Scipy
    kde = gaussian_kde(similarities)
    x_kde = np.linspace(min(similarities), max(similarities), 500)
    y_kde = kde(x_kde)

    # Plot KDE curve
    plt.plot(x_kde, y_kde * len(similarities) * (bins[1] - bins[0]), color="orange", label="KDE Curve", linewidth=2)

    # Add mean and median lines
    plt.axvline(mean_similarity, color="red", linestyle="--", label=f"Mean: {mean_similarity:.2f}")
    plt.axvline(median_similarity, color="green", linestyle="--", label=f"Median: {median_similarity:.2f}")

    # Add titles and labels
    plt.title("Document Similarity to SRP Hash Bucket Centroid with KDE (Matplotlib + Scipy)", fontsize=16)
    plt.xlabel("Cosine Similarity", fontsize=14)
    plt.ylabel("Number of Documents", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def plot_retrieval_vs_buckets(tfidf_matrix, labels, n_planes_range):
    """
    Plot retrieval precision vs. number of SRP hash buckets.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - labels: Ground truth labels for the documents.
        - n_planes_range: List or range of hyperplane counts to test.
    """
    precisions = []
    for n_planes in n_planes_range:
        # Perform SRP
        hash_codes = signed_random_projections(tfidf_matrix, n_planes)
        bucket_ids = np.dot(hash_codes, 1 << np.arange(hash_codes.shape[1]))
        
        # Evaluate precision (or any metric) based on buckets
        from sklearn.metrics import precision_score
        predicted_labels = bucket_ids  # Treat bucket IDs as pseudo-labels
        precision = precision_score(labels, predicted_labels, average="macro", zero_division=0)
        precisions.append(precision)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(n_planes_range, precisions, marker='o', color='green', label="Precision")
    plt.title("Retrieval Precision vs. Number of SRP Hyperplanes", fontsize=16)
    plt.xlabel("Number of Hyperplanes (n_planes)", fontsize=14)
    plt.ylabel("Precision (Macro Average)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.show()

def visualize_srp_with_categories(tfidf_matrix, hash_codes, labels, target_names, include_centroids=True):
    """
    Visualize SRP-LSH results with PCA and optionally plot similarity to centroids.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - hash_codes: Binary hash codes for each document.
        - labels: Ground truth category labels for each document.
        - target_names: List of category names corresponding to labels.
        - include_centroids: Whether to plot similarity to SRP centroids.
    """
    # Reduce the TF-IDF matrix to 2D using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())

    # Convert binary hash codes to integers
    hash_labels = np.dot(hash_codes, 1 << np.arange(hash_codes.shape[1]))

    # Define unique shapes for categories
    shapes = ['o', 's', 'D', '^', 'P', 'X']
    unique_categories = np.unique(labels)
    shape_mapping = {cat: shapes[i % len(shapes)] for i, cat in enumerate(unique_categories)}

    # Create scatter plot
    plt.figure(figsize=(12, 8))
    scatter = None
    for category in unique_categories:
        indices = labels == category
        scatter = plt.scatter(
            reduced_data[indices, 0], reduced_data[indices, 1],
            c=hash_labels[indices],  # Use hash labels for colors
            cmap='viridis',          # Color map for buckets
            label=target_names[category],
            alpha=0.8,
            s=100,
            marker=shape_mapping[category]
        )

    # Add colorbar for hash buckets
    cbar = plt.colorbar(scatter)
    cbar.set_label("Hash Bucket")

    # Add legend for ground truth categories
    plt.legend(title="Categories")
    plt.title("Cluster Visualization with SRP-LSH (Colors: Buckets, Shapes: Categories)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()

    # Optionally, plot similarity to SRP centroids
    if include_centroids:
        print("Plotting similarity to SRP centroids...")
        plot_similarity_to_srp_centroids(tfidf_matrix, hash_codes)

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
