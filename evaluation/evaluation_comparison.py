import numpy as np
from lsh_methods.lsh_methods import kmeans_lsh, signed_random_projections_lsh
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def compute_lsh_precisions(tfidf_matrix, categories_of_documents):
    """
    Evaluate K-means LSH and SRP LSH precisions for different numbers of clusters and planes,
    and plot the results.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - categories_of_documents: Mapping of categories to the list of document indices.
    """
    cluster_range = [3,6,9,12,15,18,21,24,27,30]
    plane_range = [3,6,9,12,15,18,21,24,27,30]

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

    for method, precision_values in results.items():
        print(f"Method: {method}")
        for param, precision in precision_values.items():
            print(f"  {param}: Precision = {precision:.4f}")

    # Extract data for plotting
    kmeans_precisions = [results["kmeans"][n] for n in cluster_range]
    srp_precisions = [results["srp"][n] for n in plane_range]


    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, kmeans_precisions, marker="o", label="K-means LSH")
    plt.plot(plane_range, srp_precisions, marker="s", label="SRP LSH")
    plt.xlabel("Number of Clusters / Planes", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision of K-means LSH vs. SRP LSH", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    comparison_plots_dir = os.path.join(os.getcwd(), "plots/comparison_plots") #setting output directory to be comparison_plots
    plt.savefig(os.path.join(comparison_plots_dir, "compute_lsh_precisions.png"))
    plt.show()
