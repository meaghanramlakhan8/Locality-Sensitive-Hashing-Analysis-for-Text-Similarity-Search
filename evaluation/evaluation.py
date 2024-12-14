import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from matplotlib import cm, colors

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

def plot_clusters(tfidf_matrix, kmeans_labels, categories_of_documents):
    """
    Visualize the clusters using PCA with colors for clusters and shapes for categories.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - kmeans_labels: Cluster labels from K-means.
        - categories_of_documents: Dictionary mapping category names to lists of document indices.
    """
    # Step 1: Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())

    # Step 2: Define marker styles for categories
    marker_styles = ['o', 's', 'D', '^']  # Circle, square, diamond, triangle
    category_names = list(categories_of_documents.keys())
    marker_map = {category: marker_styles[i % len(marker_styles)] for i, category in enumerate(category_names)}

    # Step 3: Create scatter plot for clusters
    plt.figure(figsize=(12, 8))
    for category, doc_indices in categories_of_documents.items():
        plt.scatter(
            reduced_data[doc_indices, 0],  # PCA Component 1
            reduced_data[doc_indices, 1],  # PCA Component 2
            c=[kmeans_labels[i] for i in doc_indices],  # Cluster colors
            cmap='viridis',  # Colormap for clusters
            alpha=0.7,
            marker=marker_map[category],
            label=f"{category}",
            edgecolor='k'
        )

    # Step 4: Add colorbar for clusters
    plt.colorbar(label="Cluster")

    # Step 5: Add legend for categories
    plt.legend(loc='best', title="Categories", fontsize='small')

    # Step 6: Add titles and axis labels
    plt.title("Visualization for K-Means LSH (Colors: Clusters, Shapes: Actual Categories)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()