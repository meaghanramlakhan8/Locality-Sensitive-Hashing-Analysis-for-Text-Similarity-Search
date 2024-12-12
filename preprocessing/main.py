from preprocessing import get_data, preprocess
from lsh_methods import kmeans_lsh, signed_random_projections_lsh
from evaluation import evaluate_retrieval, plot_clusters
from sklearn.metrics.pairwise import cosine_similarity

def main():
    # Step 1: Load and preprocess data
    print("Fetching and preprocessing data...")
    texts, labels, target_names = get_data(sample_size=250)
    tfidf_matrix, vectorizer = preprocess(texts)
    print("Data preprocessing completed.")
    
    # Step 2: Apply K-means LSH
    print("Applying K-means LSH...")
    kmeans_labels = kmeans_lsh(tfidf_matrix)
    plot_clusters(tfidf_matrix, kmeans_labels, title="K-means LSH Clusters")
    
    # Step 3: Apply Signed Random Projections LSH
    print("Applying Signed Random Projections LSH...")
    srp_hashes = signed_random_projections_lsh(tfidf_matrix)
    print(f"SRP Hash Codes (first 5 documents):\n{srp_hashes[:5]}")
    
    # Step 4: Evaluate retrieval metrics
    print("Evaluating K-means LSH...")
    metrics = evaluate_retrieval(labels, kmeans_labels)
    print(f"K-means LSH Metrics:\n{metrics}")
    
    # Step 5: Compute and print cosine similarity
    print("Calculating cosine similarity matrix...")
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"Cosine Similarity Matrix (first 5 rows):\n{similarity_matrix[:5, :5]}")

    # Get a list of the terms in order of frequency
    print("Ranking terms by frequency...")
    terms = vectorizer.get_feature_names_out()
    term_frequencies = tfidf_matrix.sum(axis=0).A1  # Convert to array
    term_frequency_pairs = list(zip(terms, term_frequencies))
    ranking = sorted(term_frequency_pairs, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    main()
