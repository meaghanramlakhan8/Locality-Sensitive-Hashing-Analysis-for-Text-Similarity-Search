import numpy as np
from sklearn.cluster import KMeans

def kmeans_lsh(tfidf_matrix, n_clusters=10):
    """
    Perform K-means clustering and use cluster assignments as hash codes.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - n_clusters: Number of clusters.
    
    Returns:
        - Cluster labels for each document.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(tfidf_matrix)
    return labels

def signed_random_projections_lsh(tfidf_matrix, n_planes=10):
    """
    Perform Signed Random Projections LSH.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix.
        - n_planes: Number of random hyperplanes.
    
    Returns:
        - Binary hash codes for each document.
    """
    random_planes = np.random.randn(n_planes, tfidf_matrix.shape[1])
    projections = tfidf_matrix.dot(random_planes.T)
    hash_codes = (projections > 0).astype(int)
    return hash_codes
