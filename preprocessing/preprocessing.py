from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import nltk
import re

nltk.download('stopwords')

def get_data(sample_size=None):
    """
    Load the 20 Newsgroups dataset, focusing on selected categories.

    Params:
        - sample_size (int, optional): Number of documents to load. If None, loads all data.

    Returns:
        - filtered_texts (list of str): List of preprocessed text documents (numbers removed).
        - data.target (numpy.ndarray): Numerical labels corresponding to document categories.
        - data.target_names (list of str): Names of all categories in the dataset.
    """
    #define categories for classification
    sports_categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
    science_categories = ['sci.electronics', 'sci.med', 'sci.space']
    comp_categories = ['comp.sys.mac.hardware', 'comp.graphics', 'comp.windows.x', 'comp.sys.ibm.pc.hardware']
    religion_categories = ['talk.religion.misc', 'soc.religion.christian']
    categories = sports_categories + science_categories + comp_categories + religion_categories

    #load the dataset with selected categories and remove unwanted sections
    data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

    def remove_numbers(text):
        """
        Remove all numerical content from a given text document.

        Params:
            - text (str): The input text.

        Returns:
            - (str): Text with all numbers removed.
        """
        return re.sub(r'\d+', '', text)

    #apply preprocessing step to remove numbers
    filtered_texts = [remove_numbers(doc) for doc in data.data]

    #if a sample size is specified, return only that many documents
    if sample_size:
        return filtered_texts[:sample_size], data.target[:sample_size], data.target_names

    #return all data
    return filtered_texts, data.target, data.target_names


def preprocess(texts, labels, target_names, max_features=15000, min_df=10):
    """
    Convert raw text documents into a numerical representation using TF-IDF.

    Params:
        - texts (list of str): List of raw text documents.
        - labels (numpy.ndarray): Numerical labels corresponding to document categories.
        - target_names (list of str): Names of all categories in the dataset.
        - max_features (int): Maximum number of terms to include in the TF-IDF matrix.
        - min_df (int): Minimum document frequency for terms to be included.

    Returns:
        - tfidf_matrix (scipy.sparse.csr_matrix): Sparse TF-IDF matrix representing the documents.
        - vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer with vocabulary and statistics.
        - categories_of_documents (dict): Mapping of category names to document indices.
    """
    #initialize the TF-IDF vectorizer with stopwords and settings
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, max_features=max_features, min_df=min_df)

    #fit and transform the raw text documents into the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(texts)

    #create a mapping of category names to their corresponding document indices
    categories_of_documents = defaultdict(list)
    for doc_idx, label in enumerate(labels):
        category_name = target_names[label]
        categories_of_documents[category_name].append(doc_idx)

    #display debug information (optional for understanding the dataset)
    print("")
    print("categories_of_documents: ", categories_of_documents)  # Category-to-document mapping
    print("")
    print("Vocabulary:", vectorizer.vocabulary_)  # Vocabulary created by the vectorizer
    print("")
    print("tfidf matrix: ", tfidf_matrix)  # TF-IDF matrix representation
    print("")

    return tfidf_matrix, vectorizer, categories_of_documents


def get_term_occurrences(tfidf_matrix, vectorizer):
    """
    Display the terms in the TF-IDF matrix and the documents where they appear.

    Params:
        - tfidf_matrix (scipy.sparse.csr_matrix): Sparse TF-IDF matrix representing the documents.
        - vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer with vocabulary and statistics.
    """
    #extract the terms (features) from the TF-IDF vectorizer
    terms = vectorizer.get_feature_names_out()

    #initialize a dictionary to map terms to their document indices
    term_map = defaultdict(list)

    #iterate through the rows of the TF-IDF matrix (each document)
    for i, doc in enumerate(tfidf_matrix):
        # Get the indices of non-zero terms in the document
        term_indices = doc.nonzero()[1]

        #map each term to the current document index
        for j in term_indices:
            term_map[terms[j]].append(i)

    #display each term and the documents where it appears
    for term, docs in term_map.items():
        print(f"Term: {term}")
        print(f"Appears in Documents: {docs}")
        print("")
