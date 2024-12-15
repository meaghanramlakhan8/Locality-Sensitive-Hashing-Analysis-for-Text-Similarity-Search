import numpy as np
from lsh_methods.lsh_methods import kmeans_lsh, signed_random_projections_lsh
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import defaultdict

def generate_ground_truth(labels):
    """
    Generate ground truth for Precision-Recall evaluation.

    Params:
        - labels: Ground truth category labels for each document.

    Returns:
        - y_true: Binary relevance labels for all document pairs.
    """
    n = len(labels)
    y_true = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            y_true[i, j] = 1 if labels[i] == labels[j] else 0
    return y_true.flatten()

def compute_similarity_scores(tfidf_matrix, method="srp", cluster_labels=None):
    """
    Compute similarity scores for SRP-LSH or K-means LSH.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - method: "srp" or "kmeans".
        - cluster_labels: Cluster assignments for K-means (required for "kmeans").

    Returns:
        - scores: Pairwise similarity scores for the selected method.
    """
    if method == "srp":
        # Use cosine similarity for all documents
        return cosine_similarity(tfidf_matrix)

    elif method == "kmeans" and cluster_labels is not None:
        # Compute similarity within clusters
        scores = []
        for cluster in np.unique(cluster_labels):
            # Get documents in the same cluster
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_matrix = tfidf_matrix[cluster_indices]
            cluster_scores = cosine_similarity(cluster_matrix)
            scores.append(cluster_scores.flatten())
        return np.concatenate(scores)

    else:
        raise ValueError("Invalid method or missing cluster labels for K-means.")


def compute_lsh_precisions(tfidf_matrix, categories_of_documents):
    """
    Evaluate K-means LSH and SRP LSH precisions for different numbers of clusters and planes.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - categories_of_documents: Mapping of categories to the list of document indices.
        - cluster_range: List of cluster sizes to evaluate for K-means.
        - plane_range: List of numbers of hyperplanes to evaluate for SRP.

    Returns:
        - results: Dictionary containing precision scores for both K-means and SRP.
    """
    cluster_range = [5, 10, 15, 20, 25, 30]
    plane_range = [5, 10, 15, 20, 25, 30]

    # Reverse mapping: document index -> category
    doc_to_category = {}
    for category, doc_indices in categories_of_documents.items():
        for doc_idx in doc_indices:
            doc_to_category[doc_idx] = category

    results = {"kmeans": {}, "srp": {}}

    # Evaluate K-means LSH
    for n_clusters in cluster_range:
        kmeans_labels = kmeans_lsh(tfidf_matrix, n_clusters=n_clusters)

        kmeans_clusters = defaultdict(list)
        for doc_idx, cluster in enumerate(kmeans_labels):
            category = doc_to_category.get(doc_idx, "Unknown")
            kmeans_clusters[cluster].append(category)

        precision_kmeans = 0
        for cluster, cluster_categories in kmeans_clusters.items():
            # Find the most frequent category in the cluster
            majority_label_count = max([cluster_categories.count(category) for category in set(cluster_categories)])
            precision_kmeans += majority_label_count / len(cluster_categories)

        precision_kmeans /= len(kmeans_clusters)
        results["kmeans"][n_clusters] = precision_kmeans

    # Evaluate SRP-LSH
    for n_planes in plane_range:
        srp_hashes = signed_random_projections_lsh(tfidf_matrix, n_planes=n_planes)
        srp_buckets = np.dot(srp_hashes, 1 << np.arange(srp_hashes.shape[1]))

        srp_buckets_to_categories = defaultdict(list)
        for doc_idx, bucket in enumerate(srp_buckets):
            category = doc_to_category.get(doc_idx, "Unknown")
            srp_buckets_to_categories[bucket].append(category)

        precision_srp = 0
        for bucket, bucket_categories in srp_buckets_to_categories.items():
            # Find the most frequent category in the bucket
            majority_label_count = max([bucket_categories.count(category) for category in set(bucket_categories)])
            precision_srp += majority_label_count / len(bucket_categories)

        precision_srp /= len(srp_buckets_to_categories)
        results["srp"][n_planes] = precision_srp

    return results