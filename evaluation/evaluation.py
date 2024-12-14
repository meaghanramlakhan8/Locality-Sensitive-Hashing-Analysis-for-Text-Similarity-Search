import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, silhouette_samples, silhouette_score
import numpy as np
from matplotlib import cm, colors
from sklearn.manifold import TSNE


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

def plot_by_frequency(tfidf_matrix, vectorizer, top_n=25):
    """
    Rank terms by their overall frequency across all documents and plot the top terms.

    Params:
       - tfidf_matrix : The sparse TF-IDF matrix.
       - vectorizer : The fitted TfidfVectorizer. This contains the vocabulary and statistics for the terms we have.
       - top_n : The number of top terms to display in the plot (default is 25).
    """
    # Extract terms and their frequencies
    terms = vectorizer.get_feature_names_out() 
    term_frequencies = tfidf_matrix.sum(axis=0)  # Get the sum of TF-IDF scores for each term across all documents
    term_frequencies = term_frequencies.A1  # Convert to a flat array

    # Combine terms with their frequencies
    term_frequency_pairs = list(zip(terms, term_frequencies))

    # Sort terms by frequency in descending order
    ranking = sorted(term_frequency_pairs, key=lambda x: x[1], reverse=True)

    # Display the top terms and their frequencies
    print("Top Terms by Frequency:")
    for term, frequency in ranking[:top_n]: 
        print(f"Term: {term}, Frequency: {frequency}")
    
    # Plot the top terms by frequency
    top_terms, top_frequencies = zip(*ranking[:top_n])
    plt.figure(figsize=(10, 6))
    plt.barh(top_terms, top_frequencies, color='skyblue')
    plt.gca().invert_yaxis()  # Invert y-axis to display the most frequent terms at the top
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
    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())

    # Define marker styles for categories
    marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'h', 'p', '*', 'x']  # different markers for the points
    category_names = list(categories_of_documents.keys())
    marker_map = {category: marker_styles[i % len(marker_styles)] for i, category in enumerate(category_names)}

    # Create scatter plot for clusters
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

    plt.colorbar(label="Cluster")

    plt.legend(loc='best', title="Categories", fontsize='small')

    plt.title("Visualization for K-Means LSH (Colors: Clusters, Shapes: Categories)")
    plt.xlabel("Textual Variance Dimension 1")
    plt.ylabel("Textual Variance Dimension 2")
    plt.show()

def plot_radial_clusters(kmeans_labels, categories_of_documents):
    """
    Visualize clusters in a radial layout with each cluster occupying a section of the circle.
    Points are given different shapes based on their categories.

    Params:
        - kmeans_labels: Cluster labels from K-means.
        - categories_of_documents: Dictionary mapping category names to document indices.
    """
    plt.figure(figsize=(8, 8))
    
    #Define the radial layout
    num_clusters = len(set(kmeans_labels))
    theta = np.linspace(0, 2 * np.pi, num_clusters, endpoint=False)
    cluster_colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))
    cluster_positions = {cluster: angle for cluster, angle in enumerate(theta)}

    #Define marker styles for categories
    marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'h', 'p', '*', 'x'] # different markers for the points
    category_names = list(categories_of_documents.keys())
    marker_map = {category: marker_styles[i % len(marker_styles)] for i, category in enumerate(category_names)}

    #Plot each document with cluster color and category shape
    for category, doc_indices in categories_of_documents.items():
        for doc in doc_indices:
            cluster = kmeans_labels[doc]
            angle = cluster_positions[cluster]
            x = np.cos(angle) + 0.1 * np.random.randn()  # Add noise for spread
            y = np.sin(angle) + 0.1 * np.random.randn()
            plt.scatter(
                x, 
                y, 
                color=cluster_colors[cluster], 
                marker=marker_map[category], 
                alpha=0.7, 
                edgecolor='k',
                label=category if doc_indices.index(doc) == 0 else ""  # Add label only once per category
            )

    plt.gca().set_aspect('equal', 'box')
    plt.title("Kmeans Radial Cluster Visualization (Shapes: Categories)")
    plt.legend(loc='best', title="Categories", fontsize='small')
    plt.show()

def plot_silhouette(tfidf_matrix, kmeans_labels, categories_of_documents, target_names):
    """
    Plot a silhouette graph for K-means clustering, with annotations for category contributions.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - kmeans_labels: Cluster labels from K-means.
        - categories_of_documents: Mapping of categories to document indices.
        - target_names: List of category names corresponding to category indices.
    """
    silhouette_vals = silhouette_samples(tfidf_matrix, kmeans_labels)
    silhouette_avg = silhouette_score(tfidf_matrix, kmeans_labels)
    y_lower = 10
    n_clusters = len(set(kmeans_labels))

    plt.figure(figsize=(15, 8))

    for i in range(n_clusters):
        # Get silhouette values for the ith cluster
        ith_cluster_silhouette_vals = silhouette_vals[kmeans_labels == i]
        ith_cluster_silhouette_vals.sort()

        # Identify documents in the ith cluster
        doc_indices = np.where(kmeans_labels == i)[0]

        # Count category contributions
        category_counts = {}
        for category, indices in categories_of_documents.items():
            count = len(set(indices) & set(doc_indices))  # Intersection of indices
            if count > 0:
                category_counts[category] = count

        # Get the most common category in the cluster
        predominant_category = max(category_counts, key=category_counts.get)
        category_labels = ", ".join(
            [f"{cat}: {count}" for cat, count in category_counts.items()]
        )

        # Visualize the silhouette values for the cluster
        y_upper = y_lower + len(ith_cluster_silhouette_vals)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_vals, alpha=0.7, label=f"Cluster {i} ({predominant_category})")
        plt.text(-0.05, y_lower + 0.5 * len(ith_cluster_silhouette_vals), f"Cluster {i}\n{category_labels}", fontsize=9)
        y_lower = y_upper + 10  # Add gap between clusters

    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.title("Silhouette Plot for K-means Clusters with Categories")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster")
    plt.legend(title="Cluster and Predominant Category", loc="upper right", fontsize="small")
    plt.show()

def write_clusters_to_file(kmeans_labels, categories_of_documents, output_file="kmeans_results.txt"):
    """
    Write clusters and their corresponding documents along with categories to a file.

    Params:
        - kmeans_labels: Cluster labels for each document.
        - categories_of_documents: Mapping of categories to document indices.
        - output_file: Name of the output file to write to.
    """
    #Reverse mapping to find category of a document
    doc_to_category = {}
    for category, doc_indices in categories_of_documents.items():
        for doc_idx in doc_indices:
            doc_to_category[doc_idx] = category

    #Group documents by clusters
    clusters = {}
    for doc_idx, cluster in enumerate(kmeans_labels):
        if cluster not in clusters:
            clusters[cluster] = []
        category = doc_to_category.get(doc_idx, "Unknown")
        clusters[cluster].append((doc_idx, category))

    with open(output_file, "w") as f:
        for cluster, docs in sorted(clusters.items()):
            # Count documents by category for this cluster
            category_counts = {}
            for _, category in docs:
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1

            #Write the category counts for each of the clusters 
            f.write(f"Cluster {cluster}:\n")
            f.write("  Category Counts:\n")
            for category, count in category_counts.items():
                f.write(f"    {category}: {count}\n")

            #Write document details
            f.write("  Documents:\n")
            for doc_idx, category in docs:
                f.write(f"    Document {doc_idx} - Category: {category}\n")
            f.write("\n")
    
    print(f"Clusters written to {output_file}")