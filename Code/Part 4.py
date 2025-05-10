import math
import json
import os
# import pandas as pd
# import numpy as np
from collections import Counter
# from task1 import preprocess   # Assuming you have a preprocess function in task1
from task3 import load_data, compute_tf  # Or adjust imports accordingly

def load_inverted_index(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Inverted index file '{filename}' not found")

def build_passage_tf_from_inverted_index(inverted_index):
    """
    Build a mapping from passage IDs to their term frequency dictionaries using the inverted index.
    Also build a mapping from passage IDs to the passage length (from any posting for that passage).
    """
    passage_tf_dict = {}
    passage_length_dict = {}
    for term, data in inverted_index.items():
        for pid, posting in data["postings"].items():
            # Use pid as string (as stored in the inverted index)
            if pid not in passage_tf_dict:
                passage_tf_dict[pid] = {}
                passage_length_dict[pid] = posting["length"]  # Assume this is consistent for the passage.
            passage_tf_dict[pid][term] = posting["tf"]
    return passage_tf_dict, passage_length_dict

def compute_collection_stats_from_inverted_index(inverted_index, candidate_passages_df_no_duplicates):


    # Collection term frequencies from the inverted index:
    collection_tf = {term: data["total_term_freq"] for term, data in inverted_index.items()}
    # Compute total collection words by summing lengths from unique passages.
    total_collection_words = sum(len(passage) for passage in candidate_passages_df_no_duplicates["processed"])
    return collection_tf, total_collection_words


def compute_query_likelihood(query, passage_tf, passage_length, vocab_size, smoothing, collection_tf=None, total_collection_words=None, epsilon=0.1, mu=50):
    score = 0
    query_tf = compute_tf(query, normalise=False)
    
    for term in query_tf:
        passage_term_freq = passage_tf.get(term, 0)
        query_term_freq = query_tf[term]

        if smoothing == "laplace":
            prob_t_P = (passage_term_freq + 1) / (passage_length + vocab_size)

        elif smoothing == "lidstone":
            prob_t_P = (passage_term_freq + epsilon) / (passage_length + epsilon * vocab_size)

        elif smoothing == "dirichlet":
            # Compute probability of term in collection:
            prob_t_C = collection_tf.get(term, 0) / total_collection_words  
            prob_t_P = (passage_term_freq + mu * prob_t_C) / (passage_length + mu)
        
        else:
            raise ValueError("Invalid smoothing method")

        if prob_t_P == 0:
            prob_t_P = 1e-10  # small constant to prevent log(0)

        score += query_term_freq * math.log(prob_t_P)
    return score

def rank_passage_query_likelihood(test_queries_df, candidate_passages_df, candidate_passages_df_no_duplicates, inverted_index, collection_tf, total_collection_words, smoothing, epsilon, mu):
    vocab_size = len(collection_tf)
    # Build passage term frequency and length dictionaries from the inverted index.
    passage_tf_dict, passage_length_dict = build_passage_tf_from_inverted_index(inverted_index)
    query_results = {}

    for index, row in test_queries_df.iterrows():
        qid = row["qid"]
        query_text = row["processed"]
        filter_rows = candidate_passages_df[candidate_passages_df["qid"] == qid]
        score_list = []
        for _, passage in filter_rows.iterrows():
            pid_str = str(passage["pid"])  # Ensure consistent key type.
            passage_tf = passage_tf_dict.get(pid_str, {})
            # Use passage length from the inverted index if available; otherwise fallback to computed length.
            passage_length = passage_length_dict.get(pid_str, len(passage["processed"]))
            # (Optional) You can print these values for debugging:
            # print(f"passage length from inverted index: {passage_length}")
            score = compute_query_likelihood(query_text, passage_tf, passage_length, vocab_size, smoothing, collection_tf, total_collection_words, epsilon, mu)
            score_list.append((pid_str, score))
        query_results[qid] = sorted(score_list, key=lambda x: x[1], reverse=True)[:100]

    test_queries_df["pid_ranked"] = test_queries_df["qid"].map(query_results)
    return test_queries_df

def save_file(test_queries_df, filename):
    with open(filename, "w") as f:
        for _, row in test_queries_df.iterrows():
            qid = row["qid"]
            rankings = row['pid_ranked']
            for pid, score in rankings:
                f.write(f"{qid},{pid},{score}\n")

def main():
    # Load test queries and candidate passages.
    test_queries_df, candidate_passages_df, candidate_passages_df_no_duplicates = load_data()
    # Load the inverted index (using your new structure).
    inverted_index = load_inverted_index("inverted_index.json")
    total_docs = len(candidate_passages_df_no_duplicates)
    
    # Compute collection statistics using the inverted index.
    collection_tf, total_collection_words = compute_collection_stats_from_inverted_index(inverted_index, candidate_passages_df_no_duplicates)
    
    # For the three smoothing methods, rank passages and save the results.
    test_queries_df_laplace = test_queries_df.copy()
    test_queries_df_lidstone = test_queries_df.copy()
    test_queries_df_dirichlet = test_queries_df.copy()

    # Laplace smoothing.
    test_queries_df_laplace = rank_passage_query_likelihood(test_queries_df_laplace, candidate_passages_df, candidate_passages_df_no_duplicates,
                                                            inverted_index, collection_tf, total_collection_words,
                                                            smoothing='laplace', epsilon=0.1, mu=50)
    save_file(test_queries_df_laplace, filename="laplace.csv")

    # Lidstone smoothing.
    test_queries_df_lidstone = rank_passage_query_likelihood(test_queries_df_lidstone, candidate_passages_df, candidate_passages_df_no_duplicates,
                                                             inverted_index, collection_tf, total_collection_words,
                                                             smoothing='lidstone', epsilon=0.1, mu=50)
    save_file(test_queries_df_lidstone, filename="lidstone.csv")
    
    # Dirichlet smoothing.
    test_queries_df_dirichlet = rank_passage_query_likelihood(test_queries_df_dirichlet, candidate_passages_df, candidate_passages_df_no_duplicates,
                                                             inverted_index, collection_tf, total_collection_words,
                                                             smoothing='dirichlet', epsilon=0.1, mu=50)
    save_file(test_queries_df_dirichlet, filename="dirichlet.csv")

if __name__ == "__main__":
    main()
