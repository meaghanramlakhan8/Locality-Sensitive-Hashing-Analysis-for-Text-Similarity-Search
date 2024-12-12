import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_retrieval(ground_truth, predictions):
    """
    Evaluate precision, recall, and F1 score for the retrieval.

    Params:
        - ground_truth: True labels for documents.
        - predictions: Retrieved labels for comparison.
    
    Returns:
        - Metrics dictionary containing precision, recall, and F1-score.
    """
    precision = precision_score(ground_truth, predictions, average='weighted')
    recall = recall_score(ground_truth, predictions, average='weighted')
    f1 = f1_score(ground_truth, predictions, average='weighted')
    return {'precision': precision, 'recall': recall, 'f1': f1}

def plot_top_terms(term_frequency_pairs, top_n=25):
    """
    Visualize the top terms by frequency.

    Params:
        - term_frequency_pairs: List of (term, frequency) tuples.
        - top_n: Number of top terms to display.
    """
    terms, frequencies = zip(*term_frequency_pairs[:top_n])
    plt.figure(figsize=(10, 6))
    plt.barh(terms, frequencies, color='skyblue')
    plt.gca().invert_yaxis()  # Highest frequency at the top
    plt.title(f"Top {top_n} Terms by Frequency")
    plt.xlabel("Frequency")
    plt.ylabel("Terms")
    plt.show()

def plot_clusters(tfidf_matrix, labels):
    """
    Visualize the clusters using PCA.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - labels: Cluster labels.
    """
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())
    
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(label="Cluster")
    plt.title("Cluster Visualization (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()
