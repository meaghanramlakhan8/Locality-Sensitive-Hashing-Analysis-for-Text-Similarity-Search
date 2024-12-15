import numpy as np
from sklearn.cluster import KMeans

def kmeans_lsh(tfidf_matrix, n_clusters=7):
    """
    Perform K-means clustering and use cluster assignments as hash codes.

    This function applies the K-means clustering algorithm to the given TF-IDF matrix.
    Each document is assigned to one of `n_clusters` clusters, and these cluster 
    assignments serve as "hash codes" for approximate similarity searching.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix (documents as rows, terms as columns).
        - n_clusters: Number of clusters (default is 7).

    Returns:
        - labels: Cluster labels for each document, representing the cluster 
                  each document is assigned to. These labels act as hash codes.
    """
    #initialize K-means model with the specified number of clusters and random seed 
    kmeans = KMeans(n_clusters=n_clusters, random_state=123)

    #fit the model and predict cluster labels for each document
    labels = kmeans.fit_predict(tfidf_matrix)

    # Return the cluster labels, which are used as hash codes
    return labels


def signed_random_projections_lsh(tfidf_matrix, n_planes=10):
    """
    Perform Signed Random Projections LSH (Locality-Sensitive Hashing).

    This function projects the documents in the TF-IDF matrix onto random hyperplanes.
    The sign of each projection determines a binary hash code for approximate similarity searching.

    Params:
        - tfidf_matrix: Sparse TF-IDF matrix (documents as rows, terms as columns).
        - n_planes: Number of random hyperplanes to use for hashing (default is 10).

    Returns:
        - Binary hash codes for each document:
            - Each document's hash code is a binary vector of size `n_planes`.
            - If the projection on a hyperplane is positive, the corresponding bit is 1;
              otherwise, it is 0.
    """
    #generate `n_planes` random hyperplanes of the same dimensionality as the TF-IDF features
    random_planes = np.random.randn(n_planes, tfidf_matrix.shape[1])

    #project the documents onto the hyperplanes
    projections = tfidf_matrix.dot(random_planes.T)

    #convert the projections into binary hash codes (1 for positive, 0 for negative)
    hash_codes = (projections > 0).astype(int)

    #return the binary hash codes
    return hash_codes
