from preprocessing.preprocessing import get_data, preprocess
from lsh_methods.lsh_methods import kmeans_lsh, signed_random_projections_lsh
from evaluation.evaluation import evaluate_retrieval, plot_clusters, plot_radial_clusters, plot_by_frequency, write_clusters_to_file
from evaluation.evaluation import visualize_cluster_counts
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def main():
    # Load and preprocess data !!!!
    texts, labels, target_names = get_data(sample_size=10000)
    tfidf_matrix, vectorizer, categories_of_documents = preprocess(texts, labels, target_names)
    print("Data preprocessing completed.")
    

    # Section for K-means LSH
    kmeans_labels = kmeans_lsh(tfidf_matrix)
    print("Applied K-means LSH.")
    plot_clusters(tfidf_matrix, kmeans_labels, categories_of_documents) #cluster plot for k-means
    plot_radial_clusters(kmeans_labels, categories_of_documents)    #radial clusters plot for k-means
    write_clusters_to_file(kmeans_labels, categories_of_documents) #outputs counts of categories per cluster into file
    visualize_cluster_counts(kmeans_labels, categories_of_documents) #visualization of the counts of categories per cluster 
    

    # Section for Signed Random Projections LSH
    print("Applying Signed Random Projections LSH...")
    srp_hashes = signed_random_projections_lsh(tfidf_matrix)
    print(f"SRP Hash Codes (first 5 documents):\n{srp_hashes[:5]}")
    
    # Evaluate retrieval metrics
    print("Evaluating K-means LSH...")
    metrics = evaluate_retrieval(labels, kmeans_labels)
    print(f"K-means LSH Metrics:\n{metrics}")
    
    # Compute and print cosine similarity
    print("Calculating cosine similarity matrix...")
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"Cosine Similarity Matrix (first 5 rows):\n{similarity_matrix[:5, :5]}")


    
    # Section for overall data visualizations
    plot_by_frequency(tfidf_matrix, vectorizer) #outputs the top 25 words across all data



    # This section is just to help us visualize data
    print("categories_of_documents: ", categories_of_documents)
    print("tfidf matrix: " , tfidf_matrix)
    print("kmeans_labels")
    np.set_printoptions(threshold=np.inf)
    print(kmeans_labels)

if __name__ == "__main__":
    main()
