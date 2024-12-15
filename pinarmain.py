from evaluation.evaluation_comparison import compute_lsh_precisions, compute_purity
from preprocessing.preprocessing import get_data, preprocess

def main():
    texts, labels, target_names = get_data()
    tfidf_matrix, vectorizer, categories_of_documents = preprocess(texts, labels, target_names)

    

if __name__ == "__main__":
    main()