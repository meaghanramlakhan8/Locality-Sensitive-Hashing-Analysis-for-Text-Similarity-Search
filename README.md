# **Comparative Analysis of K-means LSH and SRP-LSH for Text Similarity Search**

---

## **Description**
Locality-sensitive hashing (LSH) is an essential technique for approximate similarity search in high-dimensional spaces. This study explores and evaluates two LSH methods: **Signed Random Projections (SRP-LSH)** and **K-means LSH**, on the widely-used **20 Newsgroups Dataset** containing 18,000 text documents. The primary focus is on three key metrics:
1. **Precision**: How well similar documents are grouped.
2. **Query Processing Time**: Efficiency of document retrieval.
3. **Memory Usage**: Resource efficiency during execution.

This analysis provides actionable insights into the trade-offs between **scalability**, **accuracy**, and **efficiency**, helping researchers and developers select the most suitable LSH method for large-scale text similarity tasks.

---

## **Project Highlights**
### **Why LSH for Text Similarity?**
- **High Dimensionality**: Text vectors derived from TF-IDF representations are often sparse and high-dimensional, making similarity searches computationally expensive.
- **Approximate Similarity**: LSH algorithms enable fast, approximate similarity computations by reducing the search space into manageable subsets.
- **Scalable Retrieval**: Ideal for real-world applications involving millions of documents, such as search engines, recommendation systems, and document clustering.

---

## **Dataset and Preprocessing**
### **Dataset**
- **20 Newsgroups Dataset**:
  - ~18,000 documents.
  - Distributed across **20 categories**: Sports, Science, Computers, and Religion.

### **Preprocessing Steps**
- **Text Cleaning**:
  - Removed headers, footers, and non-content elements.
  - Lowercased all text.
  - Removed stopwords using **NLTK** and excluded numerical data.
- **TF-IDF Vectorization**:
  - Retained the top **15,000 terms** using `max_features=15000`.
  - Ignored terms appearing in fewer than **10 documents** using `min_df=10`.
  - Enabled inverse document frequency weighting (`use_idf=True`).

---

## **Implementation Details**

### **LSH Methods**
1. **Signed Random Projections (SRP-LSH)**:
   - Generated binary hash codes using **10 random hyperplanes** sampled from a normal distribution.
   - Created hash tables mapping each hash code to corresponding document IDs.
   - Optimized for speed and memory efficiency.

2. **K-means LSH**:
   - Clustered documents into **7 clusters** using the K-means algorithm.
   - Cluster labels served as hash codes for efficient retrieval.
   - Tuned for better semantic grouping but required higher computational resources.

### **Evaluation Metrics**
1. **Precision**:
   - Fraction of documents in a cluster/bucket belonging to the dominant category.
2. **Query Processing Time**:
   - Time taken to retrieve documents similar to a query.
3. **Memory Usage**:
   - Measured using Python’s `tracemalloc` library.

### **Experimental Setup**
- **Baseline Comparisons**:
  - Exact nearest neighbor search (upper bound).
  - Random retrieval (lower bound).
- **Scalability Analysis**:
  - Evaluated performance on dataset sizes ranging from **2,500 to 18,000 documents**.
- **Hyperparameter Tuning**:
  - Experimented with varying `n_planes` for SRP-LSH and `n_clusters` for K-means LSH.

---

## **Experimental Results**
### **Key Findings**
- **Precision**:
  - SRP-LSH achieved **~90%**, grouping semantically similar documents effectively.
  - K-means LSH reached **~80%**, but with better adaptability to non-uniform distributions.
- **Query Time**:
  - SRP-LSH maintained consistent, low query times (<0.05 seconds).
  - K-means LSH showed variability, peaking at **0.2 seconds** for large datasets.
- **Memory Usage**:
  - SRP-LSH required only **2.75 MB**, making it suitable for memory-constrained systems.
  - K-means LSH consumed **16.23 MB**, reflecting its overhead for cluster management.

### **Visualization**
1. **Precision Comparison**:
   - Plots showing SRP-LSH's higher precision across dataset sizes.
2. **Query Time Analysis**:
   - Line plots depicting SRP-LSH's faster performance.
3. **Memory Usage**:
   - Bar charts highlighting SRP-LSH's resource efficiency.

---

## **How to Run the Project**
### **Dependencies**
Install the required libraries:
```bash
pip install numpy scikit-learn nltk matplotlib
```

### **Running the Code**
1. Run the main script:
   ```bash
   python3 main.py
   ```

### **Outputs**
- **Plots**:
  - Stored in the plots folder
  - Grouped by comparison plots, kmeans plots, SRP plots
  - Memory Usage Bar Chart.
- **Results**:
  - Text files: `srp_results.txt` and `kmeans_results.txt`.

---

## **Future Work**
- Implement approximate K-means to reduce clustering overhead.
- Apply multi-probing strategies to improve SRP-LSH recall.
- Extend analysis to datasets with longer or more complex documents.