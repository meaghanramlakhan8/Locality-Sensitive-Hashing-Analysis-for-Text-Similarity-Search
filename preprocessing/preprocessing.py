from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import nltk

nltk.download('stopwords')

def get_data(sample_size=None):
    """
    Loads the 20 Newsgroups dataset. 
     
    Params:
        - sample_size: Number of documents to load (if sample_size=None, load all data).
    
    Returns:
        - data.data : list of raw text documents
        - data.target : array of numerical labels for each document
        - data.target_names : array containing the names of all the categories
    """
    categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']  #only including sports ones for now!!!
    data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
    
    if sample_size:
        return data.data[:sample_size], data.target[:sample_size], data.target_names
    
    return data.data, data.target, data.target_names

def preprocess(texts, labels, target_names, max_features=15000, min_df=10):
    """
    Takes raw text data and converts it into TF-IDF vectors.
    (basically numerical representations of the text documents)

    Params: 
        - texts : List of raw text documents.
        - max_features : Maximum number of terms to include (only the top ones).
        - min_df : Minimum document frequency for terms to be included.

    Returns:
      - tfidf_matrix : The sparse TF-IDF matrix 
            - The pair values underneath the Coords column represent the position in the matrix. The
            row represents the document index, and the column represents the term index (term index is determined
            by the TfidfVectorizer).
            - The numbers under the Values column are the TF-IDF values scores for that respective position.
      - vectorizer : The fitted TfidfVectorizer. This contains the vocabulary and statistics for the terms we have.
      - categories_of_documents : Mapping of the categories that we have to the list of indexes (documents) corresponding
      to it in the tfidf_matrix.
    """
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, max_features=max_features, min_df=min_df)

    tfidf_matrix = vectorizer.fit_transform(texts)

    #create a mapping of category names to their corresponding document indices
    categories_of_documents = defaultdict(list)
    for doc_idx, label in enumerate(labels):
        category_name = target_names[label]
        categories_of_documents[category_name].append(doc_idx)

    print("")
    print("categories_of_documents: ", categories_of_documents)
    print("")
    print("Vocabulary:", vectorizer.vocabulary_)
    print("")
    print("tfidf matrix: " , tfidf_matrix)
    print("")

    return tfidf_matrix, vectorizer, categories_of_documents

def get_term_occurrences(tfidf_matrix, vectorizer):
    """
    Display terms and the documents in which they occur.

    Params: 
        - tfidf_matrix : The sparse TF-IDF matrix.
        - vectorizer : The fitted TfidfVectorizer. This contains the vocabulary and statistics for the terms we have.
    """
    terms = vectorizer.get_feature_names_out()  
    term_map = defaultdict(list)  #mapping the terms to their document indices

    #iterating throught the tfidf_matrix
    for i, doc in enumerate(tfidf_matrix):
        #get only the non-zero terms for the document
        term_indices = doc.nonzero()[1]  
        #populate the occurances
        for j in term_indices:
            term_map[terms[j]].append(i)
    
    #print each term and the documents it occurs in 
    for term, docs in term_map.items():
        print(f"Term: {term}")
        print(f"Appears in Documents: {docs}")
        print("")

def get_by_frequency(tfidf_matrix, vectorizer):
    """
    Rank terms by their overall frequency across all documents.

    Params:
       - tfidf_matrix : The sparse TF-IDF matrix.
       - vectorizer : The fitted TfidfVectorizer. This contains the vocabulary and statistics for the terms we have.
    """
    terms = vectorizer.get_feature_names_out() 
    term_frequencies = tfidf_matrix.sum(axis=0)  #get the total number of documents the term occurs in
    term_frequencies = term_frequencies.A1  #convert to array

    #combine terms with their frequencies
    term_frequency_pairs = list(zip(terms, term_frequencies))

    #sort terms by frequency (descending order)
    ranking = sorted(term_frequency_pairs, key=lambda x: x[1], reverse=True)

    #display the top 25 terms 
    print("Top Terms by Frequency:")
    for term, frequency in ranking[:25]: 
        print(f"Term: {term}, Frequency: {frequency}")

def main():
    #get the data (only specifying 250 documents for now)
    texts, labels, target_names = get_data(250)  

    #preprocess 
    tfidf_matrix, vectorizer, categories_of_documents = preprocess(texts, labels, target_names)

    #display all of terms and which documents they occur in
    get_term_occurrences(tfidf_matrix, vectorizer)

    #get a list of the terms in order of frequency
    get_by_frequency(tfidf_matrix, vectorizer)

if __name__ == "__main__":
    main() 