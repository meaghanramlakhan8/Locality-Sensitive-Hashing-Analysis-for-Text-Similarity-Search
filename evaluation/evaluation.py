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
    #extract words and their frequencies
    terms = vectorizer.get_feature_names_out() 
    term_frequencies = tfidf_matrix.sum(axis=0)  #get the sum of TF-IDF scores for each term across all documents
    term_frequencies = term_frequencies.A1  #convert to a flat array

    #combine words with their frequencies
    term_frequency_pairs = list(zip(terms, term_frequencies))

    #aort words by frequency in descending order
    ranking = sorted(term_frequency_pairs, key=lambda x: x[1], reverse=True)

    #display the top words and their frequencies
    print("Top Words by Frequency:")
    for word, frequency in ranking[:top_n]: 
        print(f"Word: {word}, Frequency: {frequency}")
    
    #plotting
    top_terms, top_frequencies = zip(*ranking[:top_n])
    plt.figure(figsize=(10, 6))
    plt.barh(top_terms, top_frequencies, color='skyblue')
    plt.gca().invert_yaxis()
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
        - categories_of_documents : Mapping of the categories that we have to the list of indexes (documents) corresponding
        to it in the tfidf_matrix.
    """
    #reduce dimensionality using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())

    #different markers for the points
    marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'h', 'p', '*', 'x']  
    category_names = list(categories_of_documents.keys())
    marker_map = {category: marker_styles[i % len(marker_styles)] for i, category in enumerate(category_names)}

    plt.figure(figsize=(12, 8))
    for category, doc_indices in categories_of_documents.items():
        plt.scatter(
            reduced_data[doc_indices, 0],  # PCA Component 1
            reduced_data[doc_indices, 1],  # PCA Component 2
            c=[kmeans_labels[i] for i in doc_indices],  #colors for the clusters
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
        - categories_of_documents : Mapping of the categories that we have to the list of indexes (documents) corresponding
        to it in the tfidf_matrix.
    """
    plt.figure(figsize=(8, 8))
    num_clusters = len(set(kmeans_labels))
    theta = np.linspace(0, 2 * np.pi, num_clusters, endpoint=False)
    cluster_colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))
    cluster_positions = {cluster: angle for cluster, angle in enumerate(theta)}

    #different markers for the points
    marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'h', 'p', '*', 'x'] 
    category_names = list(categories_of_documents.keys())
    marker_map = {category: marker_styles[i % len(marker_styles)] for i, category in enumerate(category_names)}

    #plot each document with cluster color and category shape
    for category, doc_indices in categories_of_documents.items():
        for doc in doc_indices:
            cluster = kmeans_labels[doc]
            angle = cluster_positions[cluster]
            x = np.cos(angle) + 0.1 * np.random.randn()  
            y = np.sin(angle) + 0.1 * np.random.randn()
            plt.scatter(
                x, 
                y, 
                color=cluster_colors[cluster], 
                marker=marker_map[category], 
                alpha=0.7, 
                edgecolor='k',
                label=category if doc_indices.index(doc) == 0 else "" 
            )

    plt.gca().set_aspect('equal', 'box')
    plt.title("Kmeans Radial Cluster Visualization (Shapes: Categories)")
    plt.legend(loc='best', title="Categories", fontsize='small')
    plt.show()

def write_clusters_to_file(kmeans_labels, categories_of_documents, output_file="kmeans_results.txt"):
    """
    Write clusters and their corresponding documents along with categories to a file.

    Params:
        - kmeans_labels: Cluster labels for each document.
        - categories_of_documents : Mapping of the categories that we have to the list of indexes (documents) corresponding
        to it in the tfidf_matrix.
        - output_file: Name of the output file to write to.
    """
    #finding category of a document
    doc_to_category = {}
    for category, doc_indices in categories_of_documents.items():
        for doc_idx in doc_indices:
            doc_to_category[doc_idx] = category

    #grouping documents by clusters
    clusters = {}
    for doc_idx, cluster in enumerate(kmeans_labels):
        if cluster not in clusters:
            clusters[cluster] = []
        category = doc_to_category.get(doc_idx, "Unknown")
        clusters[cluster].append((doc_idx, category))

    with open(output_file, "w") as f:
        for cluster, docs in sorted(clusters.items()):
            #count documents by category for this cluster
            category_counts = {}
            for _, category in docs:
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1

            #write the category counts for each of the clusters 
            f.write(f"Cluster {cluster}:\n")
            f.write("  Category Counts:\n")
            for category, count in category_counts.items():
                f.write(f"    {category}: {count}\n")

            #write document details
            f.write("  Documents:\n")
            for doc_idx, category in docs:
                f.write(f"    Document {doc_idx} - Category: {category}\n")
            f.write("\n")
    
    print(f"Clusters written to {output_file}")

def visualize_cluster_counts(kmeans_labels, categories_of_documents):
    """
    Visualize the clusters and the count of documents grouped by overall categories.

    Params:
        - kmeans_labels: Cluster labels for each document.
        - categories_of_documents : Mapping of the categories that we have to the list of indexes (documents) corresponding
        to it in the tfidf_matrix.
    """
    sports_categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
    science_categories = ['sci.electronics', 'sci.med', 'sci.space']
    comp_categories = ['comp.sys.mac.hardware', 'comp.graphics', 'comp.windows.x', 'comp.sys.ibm.pc.hardware']
    religion_categories = ['talk.religion.misc', 'soc.religion.christian']

    #map individual categories to their overall category
    category_to_overall = {}
    for category in sports_categories:
        category_to_overall[category] = "Sports"
    for category in science_categories:
        category_to_overall[category] = "Science"
    for category in comp_categories:
        category_to_overall[category] = "Computers"
    for category in religion_categories:
        category_to_overall[category] = "Religion"

    #finding the overall category of each document
    doc_to_overall_category = {}
    for category, doc_indices in categories_of_documents.items():
        overall_category = category_to_overall.get(category, "Unknown")
        for doc_idx in doc_indices:
            doc_to_overall_category[doc_idx] = overall_category

    #group documents by clusters
    clusters = {}
    for doc_idx, cluster in enumerate(kmeans_labels):
        if cluster not in clusters:
            clusters[cluster] = []
        overall_category = doc_to_overall_category.get(doc_idx, "Unknown")
        clusters[cluster].append(overall_category)

    #count documents by overall category for each cluster
    cluster_overall_counts = {}
    for cluster, overall_categories in clusters.items():
        overall_counts = {}
        for overall_category in overall_categories:
            if overall_category not in overall_counts:
                overall_counts[overall_category] = 0
            overall_counts[overall_category] += 1
        cluster_overall_counts[cluster] = overall_counts

    clusters_sorted = sorted(cluster_overall_counts.keys())
    all_overall_categories = ["Sports", "Science", "Computers", "Religion", "Unknown"]

    #create matrix for counts: rows are clusters, columns are overall categories
    counts_matrix = []
    for cluster in clusters_sorted:
        row = [cluster_overall_counts[cluster].get(overall_category, 0) for overall_category in all_overall_categories]
        counts_matrix.append(row)

    counts_matrix = np.array(counts_matrix).T  

    #plot stacked bar chart
    x = np.arange(len(clusters_sorted)) 
    bar_width = 0.8
    bottom = np.zeros(len(clusters_sorted))

    plt.figure(figsize=(15, 8))
    for i, overall_category in enumerate(all_overall_categories):
        plt.bar(x, counts_matrix[i], bar_width, label=overall_category, bottom=bottom)
        bottom += counts_matrix[i]

    plt.xlabel("Clusters")
    plt.ylabel("Number of Documents")
    plt.title("Cluster Composition by Overall Categories")
    plt.xticks(x, [f"Cluster {c}" for c in clusters_sorted], rotation=45)
    plt.legend(title="Overall Categories", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()