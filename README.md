# Information Retrieval and Passage Ranking

This project explores core techniques in information retrieval, including text preprocessing, inverted indexing, and implementation of various ranking models such as TF-IDF, BM25, and probabilistic language models (Laplace, Lidstone, Dirichlet). The tasks involve preprocessing, frequency analysis, building an inverted index, and ranking passages for a set of queries.

---

## 🧩 Project Structure

### Task 1 – Preprocessing & Zipf's Law Validation
- **Text Cleaning**: Lowercasing, contraction expansion, punctuation & number removal, hyphen splitting, and stopword filtering (optional).
- **Tokenization**: Text converted to tokens for frequency analysis.
- **Zipf's Law Analysis**: Empirical vs. theoretical word frequency distributions evaluated using:
  - Normalized frequency plots
  - Log-log plots
  - KL-divergence to quantify deviation from Zipf’s distribution
- **Insight**: Stopword removal increases divergence from Zipf’s law; confirms the law is more stable with full vocabulary.

---

### Task 2 – Inverted Index Construction
- **Data**: Candidate passages preprocessed from `candidate-passages-top1000.tsv`
- **Indexing**: 
  - For each term: stored `doc_freq`, `total_term_freq`, and `postings` (with term frequency and passage length per passage)
- **Output**: Saved as `inverted_index.json`
- **Purpose**: Supports efficient lookup for ranking models (TF-IDF, BM25, and smoothing-based retrieval)

---

### Task 3 – Retrieval with TF-IDF and BM25
- **TF-IDF**:
  - Used cosine similarity between query and passage vectors
  - Saved ranked results to `tfidf.csv`
- **BM25**:
  - Used parameters: `k1=1.2`, `k2=100`, `b=0.75`
  - Saved results to `bm25.csv`
- **Output**: Both models returned top 100 passages for each query

---

### Task 4 – Query Likelihood with Smoothing
- **Models Implemented**:
  - **Laplace Smoothing**
  - **Lidstone Smoothing** (ε = 0.1)
  - **Dirichlet Smoothing** (μ = 50)
- **Result Files**: `laplace.csv`, `lidstone.csv`, `dirichlet.csv`
- **Findings**:
  - **Dirichlet** performed best by incorporating global corpus statistics
  - **Laplace** and **Lidstone** over-smoothed the scores, especially for rare terms
  - Log-likelihood scores clearly demonstrated Dirichlet’s effectiveness in relevant passage ranking

---

## 🛠 Technologies Used

- Python
- JSON, TSV parsing
- NumPy, SciPy
- Matplotlib (for plotting Zipf analysis)

---

## ✅ Key Skills Demonstrated

- Text preprocessing for NLP tasks
- Word frequency analysis and validation of Zipf’s Law
- Inverted index construction and optimization
- Implementation and comparison of multiple retrieval models
- Statistical modeling with smoothing techniques
- Efficient data structuring for fast passage scoring

