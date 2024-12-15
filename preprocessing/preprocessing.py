from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import nltk
import re

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
    sports_categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']  
    science_categories = ['sci.electronics', 'sci.med', 'sci.space', 'sci.crypt']
    comp_categories = ['comp.sys.mac.hardware', 'comp.graphics', 'comp.windows.x' , 'comp.sys.ibm.pc.hardware', 'comp.os.ms-windows.misc']
    religion_categories = ['talk.religion.misc', 'soc.religion.christian']
    categories = sports_categories + science_categories + comp_categories + religion_categories
    data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

    def remove_numbers(text):
        """
        Remove all numerical content from a given text document.
        """
        return re.sub(r'\d+', '', text)
    
    filtered_texts = [remove_numbers(doc) for doc in data.data]
    
    if sample_size:
        return filtered_texts[:sample_size], data.target[:sample_size], data.target_names
    
    return filtered_texts, data.target, data.target_names

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
