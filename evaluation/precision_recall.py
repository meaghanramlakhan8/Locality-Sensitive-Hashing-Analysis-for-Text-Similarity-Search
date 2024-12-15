import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

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
        #use cosine similarity for all documents
        return cosine_similarity(tfidf_matrix)

    elif method == "kmeans" and cluster_labels is not None:
        #compute similarity within clusters
        scores = []
        for cluster in np.unique(cluster_labels):
            #get documents in the same cluster
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_matrix = tfidf_matrix[cluster_indices]
            cluster_scores = cosine_similarity(cluster_matrix)
            scores.append(cluster_scores.flatten())
        return np.concatenate(scores)

    else:
        raise ValueError("Invalid method or missing cluster labels for K-means.")

def plot_comparative_precision_recall(y_true, scores_srp, scores_kmeans):
    """
    Plot Precision-Recall Curves for SRP-LSH and K-means LSH.

    Params:
        - y_true: Ground truth binary relevance labels (1 for relevant, 0 for non-relevant).
        - scores_srp: Predicted similarity scores for SRP-LSH.
        - scores_kmeans: Predicted similarity scores for K-means LSH.
    """
    #precision-Recall for SRP-LSH
    precision_srp, recall_srp, _ = precision_recall_curve(y_true, scores_srp)
    pr_auc_srp = auc(recall_srp, precision_srp)

    #precision-Recall for K-means LSH
    precision_kmeans, recall_kmeans, _ = precision_recall_curve(y_true, scores_kmeans)
    pr_auc_kmeans = auc(recall_kmeans, precision_kmeans)

    #plot the curves
    plt.figure(figsize=(10, 6))
    plt.plot(recall_srp, precision_srp, label=f"SRP-LSH (AUC = {pr_auc_srp:.2f})", color="blue", linewidth=2)
    plt.plot(recall_kmeans, precision_kmeans, label=f"K-means LSH (AUC = {pr_auc_kmeans:.2f})", color="orange", linewidth=2)

    plt.title("Precision-Recall Curves: SRP-LSH vs. K-means LSH", fontsize=16)
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
