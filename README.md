# **Comparative Analysis of K-means LSH and SRP-LSH for Text Similarity Search**

## **Overview**
Locality-Sensitive Hashing (LSH) is a widely-used technique for efficient similarity search in high-dimensional spaces, such as text document vectors represented by TF-IDF. While Signed Random Projections (SRP-LSH) is computationally efficient and scalable, its randomness often fails to fully exploit the inherent structure of text corpora. 

This study introduces and evaluates **K-means LSH**, which integrates clustering into the hashing process, adapting to the data distribution and potentially improving retrieval accuracy. By comparing K-means LSH to SRP-LSH, we explore the trade-offs between accuracy, efficiency, and scalability for text similarity search.

Our findings aim to guide the selection of appropriate LSH methods for applications such as semantic clustering, nearest neighbor search, and large-scale text similarity analysis.

---

## **Objectives**
- **Evaluate and compare two LSH methods** for text similarity search:
  - **Signed Random Projections (SRP-LSH)**: Uses random hyperplanes to generate binary hash codes.
  - **K-means LSH**: Uses cluster assignments from K-means clustering as hash codes.
- Analyze trade-offs between:
  - **Retrieval accuracy (Precision@k)**
  - **Indexing and query efficiency**
  - **Scalability for large datasets**
- Explore the impact of hyperparameters such as:
  - Number of hash functions (`n`) for SRP-LSH.
  - Number of clusters (`k`) for K-means LSH.
- Visualize results to provide insights into LSH performance under different conditions.

---

## **Methodology**

### **Dataset**
We utilized the **20 Newsgroups Dataset**, a collection of 18,000 posts across 20 topics. The dataset was divided into:
1. **Training data** for building LSH indices.
2. **Testing data** for evaluating retrieval performance.

### **Preprocessing**
- Removed headers, footers, and non-content elements.
- Normalized text to lowercase and removed stopwords using NLTK.
- Converted text to numerical vectors using **TF-IDF Vectorization** with the following parameters:
  - `max_features=`: Retain the top 15,000 terms.
  - `min_df=`: Ignore terms appearing in fewer than 10 documents.
  - Default `use_idf=True`: Apply inverse document frequency reweighting.

### **Techniques**
1. **Signed Random Projections LSH**:
   - Generated binary hash codes using random hyperplanes.
   - Experimented with varying numbers of hyperplanes (`n`).
   - Compared retrieval performance across settings.

2. **K-means LSH**:
   - Applied K-means clustering to TF-IDF vectors.
   - Used cluster assignments as hash codes.
   - Evaluated different numbers of clusters (`k`).

3. **Baselines**:
   - **Exact Nearest Neighbor Search** (Upper bound): Compared query vectors with all dataset vectors using cosine similarity.
   - **Random Retrieval** (Minimal baseline): Randomly selected documents for comparison.

---

## **Experiments**

### **Evaluation Metrics**
- **Precision@k**: Fraction of relevant documents among the top `k` retrieved.
- **Query Processing Time**: Time required to retrieve similar documents.
- **Memory Usage**: Amount of memory consumed by LSH indices.

### **Experimental Procedure**
1. **Hyperparameter Tuning**:
   - Varied `n` (number of hash functions) for SRP-LSH.
   - Varied `k` (number of clusters) for K-means LSH.
   - Identified optimal parameters for retrieval performance.
2. **Baseline Comparison**:
   - Evaluated both LSH methods against exact nearest neighbor search and random retrieval.
3. **Dimensionality Reduction**:
   - Applied **Truncated SVD** to analyze the impact of reducing vector dimensions on LSH performance.

---

## **Results**

### **Preliminary Findings**
1. **K-means LSH**:
   - Higher **Precision@k** than SRP-LSH, particularly for smaller datasets.
   - Longer indexing time due to clustering but better semantic grouping.
   - Increased memory usage due to cluster information storage.

2. **SRP-LSH**:
   - Faster index build time and better scalability for larger datasets.
   - Lower memory usage compared to K-means LSH.
   - Lower **Precision@k**, likely due to the randomized nature of hyperplane projections.

3. **Hyperparameter Insights**:
   - Optimal performance observed at:
     - **12 hyperplanes** for SRP-LSH.
     - **20 clusters** for K-means LSH.
   - Multi-probing strategies improved recall for SRP-LSH.

### **Visualizations**
- **Similarity Distribution**:
  - Histograms of document similarities with KDE curves for each method.
- **Cluster Analysis**:
  - PCA-based scatter plots to visualize K-means clusters and SRP hash buckets.
- **Hyperparameter Impact**:
  - Line plots showing how increasing `n` or `k` affects retrieval metrics.

---

## **Conclusion**
This study highlights the trade-offs between **K-means LSH** and **SRP-LSH** for text similarity search:
- K-means LSH excels in semantic grouping but is resource-intensive.
- SRP-LSH is faster and more scalable but less accurate for small, semantically diverse datasets.

By tuning hyperparameters and employing dimensionality reduction, both methods can achieve significant improvements in retrieval accuracy and efficiency.

---

## **Future Work**
- Implement approximate K-means clustering to reduce index build time.
- Explore multi-probing techniques to enhance SRP-LSH recall.
- Extend the analysis to other text similarity datasets and longer documents.
- Investigate hybrid methods that combine SRP-LSH and K-means clustering.

---

## **References**
- Paulevé, Loïc, et al. "Locality Sensitive Hashing: A Comparison of Hash Function Types and Querying Mechanisms." *Pattern Recognition Letters*, 2010.  
- Jafari, Omid, et al. "A Survey on Locality Sensitive Hashing Algorithms and Their Applications." *arXiv*, 2021.