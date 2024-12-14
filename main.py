from preprocessing.preprocessing import get_data, preprocess
from lsh_methods.lsh_methods import kmeans_lsh, signed_random_projections_lsh
from evaluation.evaluation import evaluate_retrieval, plot_clusters, plot_radial_clusters, plot_by_frequency, plot_silhouette, write_clusters_to_file
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def main():
    #Load and preprocess data !!!!
    print("Fetching and preprocessing data...")
    texts, labels, target_names = get_data(sample_size=5000)
    tfidf_matrix, vectorizer, categories_of_documents = preprocess(texts, labels, target_names)
    print("Data preprocessing completed.")
    
    #Section for K-means LSH
    print("Applying K-means LSH...")
    kmeans_labels = kmeans_lsh(tfidf_matrix)
    print("kmeans_labels")
    print(kmeans_labels)
    plot_clusters(tfidf_matrix, kmeans_labels, categories_of_documents)
    plot_radial_clusters(kmeans_labels, categories_of_documents)
    plot_silhouette(tfidf_matrix, kmeans_labels, categories_of_documents, target_names)
    
    #Section for Signed Random Projections LSH
    print("Applying Signed Random Projections LSH...")
    srp_hashes = signed_random_projections_lsh(tfidf_matrix)
    print(f"SRP Hash Codes (first 5 documents):\n{srp_hashes[:5]}")
    
    #Evaluate retrieval metrics
    print("Evaluating K-means LSH...")
    metrics = evaluate_retrieval(labels, kmeans_labels)
    print(f"K-means LSH Metrics:\n{metrics}")
    
    #Compute and print cosine similarity
    print("Calculating cosine similarity matrix...")
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"Cosine Similarity Matrix (first 5 rows):\n{similarity_matrix[:5, :5]}")

    #Get a list of the terms in order of frequency and plot
    plot_by_frequency(tfidf_matrix, vectorizer)

    write_clusters_to_file(kmeans_labels, categories_of_documents)

    #this section is to just try and visualize
    print("")
    print("categories_of_documents: ", categories_of_documents)
    print("")
    print("tfidf matrix: " , tfidf_matrix)
    print("")
    print("kmeans_labels")
    np.set_printoptions(threshold=np.inf)
    print(kmeans_labels)
    print(len(kmeans_labels))

if __name__ == "__main__":
    main()
