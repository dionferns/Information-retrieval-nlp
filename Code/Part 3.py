import math
import json
import os
import pandas as pd
import numpy as np
from collections import Counter
from task1 import preprocess

remove_stopwords = False

def load_inverted_index(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Inverted index file '{filename}' not found")

def load_data():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    test_queries_path = os.path.join(current_dir, "test-queries.tsv")
    candidate_passages_path = os.path.join(current_dir, "candidate-passages-top1000.tsv")

    test_queries_df = pd.read_csv(test_queries_path, sep="\t", header=None, names=['qid', 'query'])
    candidate_passages_df = pd.read_csv(candidate_passages_path, sep="\t", header=None, names=['qid', 'pid', 'query', 'passage'])
    candidate_passages_df_no_duplicates = candidate_passages_df.drop_duplicates(subset=['pid'], keep='first').copy()

    test_queries_df['processed'] = test_queries_df['query'].apply(lambda x: preprocess(x, remove_stopwords))
    candidate_passages_df['processed'] = candidate_passages_df['passage'].apply(lambda x: preprocess(x, remove_stopwords))
    candidate_passages_df_no_duplicates['processed'] = candidate_passages_df_no_duplicates['passage'].apply(lambda x: preprocess(x, remove_stopwords))

    return test_queries_df, candidate_passages_df, candidate_passages_df_no_duplicates

def compute_tf(document, normalise=True):
    total_word_count = len(document)
    term_counts = Counter(document)
    if normalise:
        tf = {word: term_counts[word] / total_word_count for word in term_counts}
    else:
        tf = dict(term_counts)
    return tf

def compute_idf(inverted_index, total_docs):
    # Standard IDF using stored doc frequency from the inverted index.
    return {word: math.log(total_docs / (1 + inverted_index[word]["doc_freq"])) 
            for word in inverted_index}

def compute_tfidf_from_tf(tf, idf):
    return {word: tf[word] * idf.get(word, 0) for word in tf}

def cosine_similarity(vec1, vec2):
    common_words = set(vec1.keys()) & set(vec2.keys())
    if not common_words:
        return 0.0
    vec1_values = np.array([vec1[word] for word in common_words])
    vec2_values = np.array([vec2[word] for word in common_words])
    dot_product = np.dot(vec1_values, vec2_values)
    norm1 = np.linalg.norm(list(vec1.values()))
    norm2 = np.linalg.norm(list(vec2.values()))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0


def compute_bm25(query_tokens, pid, passage_length, avg_doc_length, idf, k1, b, k2, inverted_index):
    score = 0.0
    pid_str = str(pid)  # Convert the passage ID to a string
    query_tf = Counter(query_tokens)
    # Iterate over unique query terms to avoid double counting.
    for term, qf in query_tf.items():
        if term in inverted_index and pid_str in inverted_index[term]["postings"]:
            tf = inverted_index[term]["postings"][pid_str]["tf"]
        else:
            tf = 0
        if tf == 0:
            continue
        term_idf = idf.get(term, 0)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (passage_length / avg_doc_length))
        query_weight = (qf * (k2 + 1)) / (qf + k2)
        score += term_idf * (numerator / denominator) * query_weight
    return score

def build_passage_tf_from_inverted_index(inverted_index):
    """
    Build a mapping from passage IDs to their term frequency dictionaries using the inverted index.
    Also build a mapping from passage IDs to the passage length (as stored in the postings).
    """
    passage_tf_dict = {}
    passage_length_dict = {}
    for word, data in inverted_index.items():
        for pid, posting in data["postings"].items():
            if pid not in passage_tf_dict:
                passage_tf_dict[pid] = {}
                passage_length_dict[pid] = posting["length"]  # Assume length is consistent across words.
            passage_tf_dict[pid][word] = posting["tf"]
    return passage_tf_dict, passage_length_dict


def rank_documents_bm25(test_queries_df, candidate_passages_df, candidate_passages_df_no_duplicates,
                        avg_doc_length, idf, k1, b, k2, inverted_index):
    # For each query, rank candidate passages using BM25.
    if "pid_ranked" not in test_queries_df.columns:
        test_queries_df["pid_ranked"] =  [[] for _ in range(len(test_queries_df))]

    for index, row in test_queries_df.iterrows():
        filtered_rows = candidate_passages_df[candidate_passages_df["qid"] == row["qid"]]
        scores_list = []
        for _, r in filtered_rows.iterrows():
            pid = r["pid"]

            score = compute_bm25(row["processed"], pid, len(r["processed"]), avg_doc_length,idf, k1, b, k2, inverted_index)

            # If the score is zero, assign a very small non-zero value.
            if score == 0:
                score = 0.0001
            scores_list.append((pid, score))
        scores_sorted = sorted(scores_list, key=lambda x: x[1], reverse=True)[:min(100, len(scores_list))]
        test_queries_df.at[index, "pid_ranked"] = scores_sorted
    return test_queries_df

def save_bm25_results(test_queries_df, filename):
    with open(filename, "w") as f:
        for _, row in test_queries_df.iterrows():
            qid = row["qid"]
            rankings = row["pid_ranked"]
            for pid, score in rankings:
                f.write(f"{qid},{pid},{score}\n")

def rank_documents_tf_idf(test_queries_df, candidate_passages_df, candidate_passages_df_no_duplicates,
                          idf, passage_tf_dict, passage_length_dict):
    # Build TF-IDF vectors for passages using term counts from the inverted index.
    passage_tfidf = {}
    for pid, tf_dict in passage_tf_dict.items():
        total_length = passage_length_dict[pid]
        # Normalize the term frequencies.
        normalised_tf = {word: count / total_length for word, count in tf_dict.items()}
        passage_tfidf[pid] = {word: normalised_tf[word] * idf.get(word, 0) for word in normalised_tf}
    
    test_queries_df["pid_ranked"] = [[] for _ in range(len(test_queries_df))]
    query_tfidf = {}
    for index, row in test_queries_df.iterrows():
        qid = row["qid"]
        # Compute the query TF and then its TF-IDF vector.
        query_tf = compute_tf(row["processed"], normalise=True)
        query_vec = compute_tfidf_from_tf(query_tf, idf)
        query_tfidf[qid] = query_vec
        
        cosine_values = []
        filtered_df = candidate_passages_df[candidate_passages_df['qid'] == qid]
        for pid in filtered_df['pid'].tolist():
            pid_str = str(pid)
            if pid_str in passage_tfidf:
                similarity = cosine_similarity(query_vec, passage_tfidf[pid_str])
                cosine_values.append((pid_str, similarity))
        cosine_values_sorted = sorted(cosine_values, key=lambda x: x[1], reverse=True)[:100]
        test_queries_df.at[index, "pid_ranked"] = list(cosine_values_sorted)
    return test_queries_df

def save_tfidf_results(test_queries_df, filename="tfidf.csv"):
    with open(filename, "w") as f:
        for _, row in test_queries_df.iterrows():
            qid = row["qid"]
            rankings = row['pid_ranked']
            for pid, score in rankings:
                f.write(f"{qid},{pid},{score}\n")

def main():
    # Load test queries and candidate passages.
    test_queries_df, candidate_passages_df, candidate_passages_df_no_duplicates = load_data()
    # Load the inverted index (built using your build_inverted_index code).
    inverted_index = load_inverted_index("inverted_index.json")
    total_docs = len(candidate_passages_df_no_duplicates)
    
    # Compute BM25-specific IDF values.
    # bm25_idf = compute_bm25_idf(inverted_index, total_docs)
    # For TF-IDF ranking, you may still use standard IDF.
    standard_idf = compute_idf(inverted_index, total_docs)
    
    # Build passage term frequencies and lengths from the inverted index.
    passage_tf_dict, passage_length_dict = build_passage_tf_from_inverted_index(inverted_index)

    # Rank documents using the TF-IDF model.
    test_queries_df_tfidf = test_queries_df.copy()
    test_queries_df_tfidf = rank_documents_tf_idf(test_queries_df_tfidf, candidate_passages_df, candidate_passages_df_no_duplicates,
                                                 standard_idf, passage_tf_dict, passage_length_dict)
    save_tfidf_results(test_queries_df_tfidf)
    
    # Compute average document length from the passages.
    # avg_doc_length = np.mean(list(passage_length_dict.values()))


    avg_doc_length = candidate_passages_df_no_duplicates["processed"].apply(len).mean()
    print("Average document length:", avg_doc_length)
    
    # Rank documents using the BM25 model.
    test_queries_df_bm25 = test_queries_df.copy()
    test_queries_df_bm25 = rank_documents_bm25(test_queries_df_bm25, candidate_passages_df, candidate_passages_df_no_duplicates,
                                               avg_doc_length, standard_idf, k1=1.2, b=0.75, k2=100,
                                               inverted_index=inverted_index)

    save_bm25_results(test_queries_df_bm25, filename="bm25.csv")

if __name__ == "__main__":
    main()