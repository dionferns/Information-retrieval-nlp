import pandas as pd
import os
from task1 import preprocess
import re
import itertools
import textwrap
import json



def load_passages(remove_stopwords):
    current_dir = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(current_dir,  "candidate-passages-top1000.tsv")
    
    df = pd.read_csv(filename, sep='\t', header=None, names=['qid', 'pid', 'query', 'passage'])     #outputs a dataframe.
    df_dropped_duplicates = df.drop_duplicates(subset=['pid'], keep='first').copy()

    df['processed'] = df['passage'].apply(lambda x: preprocess(x, remove_stopwords))
    df_dropped_duplicates['processed'] = df_dropped_duplicates['passage'].apply(lambda x: preprocess(x, remove_stopwords))

    return df, df_dropped_duplicates




def build_inverted_index(df):
    inverted_index = {}

    for _, row in df.iterrows():
        pid_value = row['pid']
        line = row['processed']
        passage_length = len(line)  # Passage length = total words in passage

        for word in line:
            if word not in inverted_index:
                inverted_index[word] = {
                    "doc_freq": 0,
                    "total_term_freq": 0,
                    "postings": {}
                }

            if pid_value not in inverted_index[word]["postings"]:
                inverted_index[word]["doc_freq"] += 1
                inverted_index[word]["postings"][pid_value] = {"tf": 0, "length": passage_length}

            # Update term frequency
            inverted_index[word]["postings"][pid_value]["tf"] += 1
            inverted_index[word]["total_term_freq"] += 1
    return inverted_index


def dict_slicing(inverted_index):
    for key, value in itertools.islice(inverted_index.items(), 30):
        print(f"{key}: {len(value)}")


def save_inverted_index(inverted_index, filename):
    """Save the inverted index to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, indent=4)

def query_inverted_index(word, inverted_index):
    """Retrieve passage IDs containing a given word."""
    return inverted_index.get(word, [])


# Loading the JSON file of inverted index.
def load_inverted_index(filename):
    """Load the inverted index from a JSON file if it exists."""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return None


def main():
    filename="inverted_index.json"    # print(texting.head(5))
    remove_stopwords=False
    
    df, df_dropped_duplicates = load_passages(remove_stopwords)

    invert_index = build_inverted_index(df_dropped_duplicates)
    save_inverted_index(invert_index, filename)
    print(f" Inverted Index Saved as '{filename}'")
    

if __name__ == "__main__":
    main()

