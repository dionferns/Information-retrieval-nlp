import collections
import os
import numpy as np
import matplotlib.pyplot as plt

import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def load_text():
    current_dir = os.path.abspath(os.path.dirname(__file__))  # Ensure absolute path

    file_path = os.path.join(current_dir, "passage-collection.txt")
    with open(file_path, 'r', encoding='utf-8') as file:
        text2 = file.read()
    return text2


'''
Question 2: Report the size of the identified index of terms (vocabulary) [1 mark]

The size of the identitfied index of terms is 120834.

'''



#  Contractions dictionary
contractions = {
    "don't": "do not", "didn't": "did not", "doesn't": "does not", "can't": "cannot", "isn't": "is not",
    "it's": "it is", "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "we're": "we are", "they're": "they are", "i've": "i have", "you've": "you have",
    "we've": "we have", "they've": "they have", "i'll": "i will", "you'll": "you will",
    "we'll": "we will", "they'll": "they will", "there's": "there is", "that's": "that is",
    "shouldn't": "should not", "won't": "will not", "wouldn't": "would not", 
    "haven't": "have not", "hasn't": "has not", "weren't": "were not",
    "mustn't": "must not", "ain't": "is not", "couldn't": "could not",
    "let's": "let us", "y'all": "you all", "shan't": "shall not"
}

def expand_contractions(text):
    """Expand contractions like don't → do not."""
    for contraction, full_form in contractions.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', full_form, text)
    return text

def preprocess(text, remove_stopwords=False):
    """Clean, tokenize, and preprocess text for general search engines."""
    
    # Convert to lowercase
    text = text.lower()

    # Expand contractions
    text = expand_contractions(text)

    # Remove ALL apostrophes (e.g., O'Connor → OConnor)
    text = re.sub(r"'", "", text)  

    # Convert hyphens into spaces (e.g., real-time → real time)
    text = re.sub(r"-", " ", text)

    # Remove everything except words and spaces
    text = re.sub(r"[^\w\s]", " ", text)  # This removes punctuation including remaining special characters

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize (split into words)
    words = text.split()

    # Remove stopwords if enabled
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))  
        words = [word for word in words if word not in stop_words]
    return words

def counter_fre(words):
    # Function counts the number of occurrences of terms in the provided data set.
    counter_words = collections.Counter(words)
    counter_words_ordered = dict(sorted(counter_words.items(), key=lambda x: x[1], reverse=True))
    return counter_words_ordered


def normalised_freq(counter_words_ordered ):
    counter_words_items = counter_words_ordered.items()
    counter_sum = sum(counter_words_ordered.values())
    norm_freq = {word: count / counter_sum for word, count in counter_words_items}

    return norm_freq

def normalised_freq_plot(norm_frequ,remove_stopwords):
    rank = np.arange(1, len(norm_frequ)+ 1)
    values1  = list(norm_frequ.values())
    plt.figure(figsize=(8, 6))  # Ensure new figure for each plot

    plt.plot(rank, values1, marker=".", linestyle='none', markersize=3, label="Emperical Word Frequency Distribution")
    plt.xlabel("Rank")
    plt.ylabel("Normalised Frequency")
    plt.title("Normalised frequency against their frequency ranking")
    plt.legend()
    if remove_stopwords:
        plt.savefig("Figure_3.1(removed_stop_words).pdf", format="pdf")  # Save the plot as an image.
    else:
        plt.savefig("Figure_1.pdf", format="pdf")  # Save the plot as an image.
    plt.close()



def zipslaw_normalised(norm_freq, remove_stopwords):
    sorted_counts = sorted(norm_freq.values(), reverse=True)
    ranks = np.arange(1, len(sorted_counts) + 1)
    norm_freq_list = list(norm_freq.values())
    plt.figure(figsize=(8, 6))

    plt.loglog(ranks, sorted_counts, marker="o", linestyle='none', markersize=3, label="Empirical Data")
    
    # Compute Zipf's law theoretical curve efficiently
    s = 1  # Zipf’s law parameter (default is 1 for natural language)
    N = len(sorted_counts)
    
    normalization = sum(float(i) ** -s for i in range(1, N + 1))  # Compute normalization only once
    zipf_values = np.array([float(k) ** -s for k in ranks]) / normalization  # Convert k to float

    zipf_sum = np.sum(zipf_values)
    print(f" Zipf distribution sum check: {zipf_sum:.6f}")  # Expect ≈ 1.000000


    plt.loglog(ranks, zipf_values, linestyle="--", label="Zipf's Law")
     
    plt.xlabel("Rank")
    plt.ylabel("Normalised Frequency")
    plt.title("Log-Log Plot of Empirical Data vs. Zipf's Law")
    plt.legend()
    if remove_stopwords:
        plt.savefig("Figure_3.pdf", format="pdf")  # Save the plot as an image.
    else:
        plt.savefig("Figure_2.pdf", format="pdf")  # Save the plot as an image.
    plt.close()


    plt.figure(figsize=(8, 6))
    plt.plot(ranks, norm_freq_list, marker=".", linestyle='none', markersize=3, label="Empirical Data")
    plt.plot(ranks, zipf_values, marker=".", linestyle='none', markersize=3, label="Zipf's Law")
    plt.xlabel("Rank")
    plt.ylabel("Normalised Frequency")
    plt.title("Emperical Word Frequency Distribution vs Theoretical Zipf's Law")
    plt.legend()

    if remove_stopwords:
        plt.savefig("(RSW)Emperical_Word_Frequency_Distribution_vs_Theoretical_Zipfs_Law.pdf", format="pdf")  # Save the plot as an image.
    else:
        plt.savefig("Emperical_Word_Frequency_Distribution_vs_Theoretical_Zipfs_Law.pdf", format="pdf")  # Save the plot as an image.
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(ranks, zipf_values, marker=".", linestyle='none', markersize=3, label="Zipf's Law")
    plt.xlabel("Rank")
    plt.ylabel("Normalised Frequency")
    plt.title("Theoretical Zipf's Law Distribution of Word Frequencies")
    plt.legend()
    
    if remove_stopwords:
        plt.savefig("(RSW)Theoretical_Zipfs_Law.pdf", format="pdf")  # Save the plot as an image.
    else:
        plt.savefig("Theoretical_Zipfs_Law.pdf", format="pdf")  # Save the plot as an image.

    plt.close()
    return sorted_counts, zipf_values 

def KL_divergence(empirical_counts, zipf_counts):
    empirical_prob = np.array(empirical_counts)
    zipf_prob = np.array(zipf_counts)
    # Check if they are already normalized (sum ≈ 1)
    empirical_sum = np.sum(empirical_prob)
    zipf_sum = np.sum(zipf_prob)

    print(f"Empirical Sum: {empirical_sum:.6f} | Zipf Sum: {zipf_sum:.6f}")
    empirical_prob = np.where(empirical_prob == 0, 1e-10, empirical_prob)
    zipf_prob = np.where(zipf_prob == 0, 1e-10, zipf_prob)

    kl_divergence = np.sum(empirical_prob * np.log(empirical_prob / zipf_prob))
    return kl_divergence


def print_most_common(counter_words, top_n=200):
    """Prints the top N most common words and their frequencies."""
    print(f"\nTop {top_n} most common words:\n")
    most_common_words = list(counter_words.items())[:top_n]
    
    for rank, (word, freq) in enumerate(most_common_words, start=1):
        print(f"{rank}. {word}: {freq}")

def main():
    # With stopwords.
    print("With stop words")
    text = load_text()
    words = preprocess(text)
    print(f"size of text: {len(words)}")
    counter_words = counter_fre(words)
    normalised_frequency = normalised_freq(counter_words)

    normalised_freq_plot(normalised_frequency, False)
    print(f"Vocabulary size: {len(counter_words)} unique words") # Report vocabulary size
    empirical_counts, zipf_counts = zipslaw_normalised(normalised_frequency ,False)
    kl_value = KL_divergence(empirical_counts, zipf_counts)
    print(f"The KL Divergence values with stop words is: {kl_value}")

    print()
    print()

    print("Without removing stopwords")
    # Removed stopwords
    text_no_sw = load_text()
    words_no_sw = preprocess(text_no_sw, remove_stopwords=True)
    print(f"size of text: {len(words_no_sw)}")
    counter_words_no_sw = counter_fre(words_no_sw)
    normalised_frequency_no_sw = normalised_freq(counter_words_no_sw)

    normalised_freq_plot(normalised_frequency_no_sw, True)
    print(f"Vocabulary size: {len(counter_words_no_sw)} unique words") # Report vocabulary size
    empirical_counts_no_sw, zipf_counts_no_sw = zipslaw_normalised(normalised_frequency_no_sw, True)

    kl_value_no_sw = KL_divergence(empirical_counts_no_sw, zipf_counts_no_sw)
    print(f"The KL Divergence values without stop words is: {kl_value_no_sw}")

    # Load the TSV file
    # df1 = pd.read_csv("candidate-passages-top1000.tsv", sep="\t")
    # df2 = pd.read_csv("passage-collection.txt", sep="\t")
    # df3 = pd.read_csv("passage-collection.txt", sep="\t")

    # Print the number of rows
    # print("Number of rows:", len(df1))
    # print("Number of rows:", len(df2))

if __name__ == "__main__":
    main()