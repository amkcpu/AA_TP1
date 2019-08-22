# Naive Bayes Text classifier
import numpy as np
import pandas as pd
import re  # for regexp splitting words
import codecs  # for unicode file reading
import os  # for looping over files in directory
import math  # for NaN checking


# Returns number of documents that a given word is found in
# Requires: word_list (containing words in rows, each document is one column), word (to be searched for)
# Important for TF-IDF
def no_documents_contain(word_list, word):
    counter = 0
    no_examples, no_documents = word_list.shape

    for i in range(0, no_documents):
        if word in word_list.iloc[:, i].values:  # tests whether the word appears in a given column i
            counter += 1

    return counter


# Term Frequency-Inverse Document Frequency (TF-IDF) algorithm
# Returns pandas.DataFrame (rows: words, columns: documents) with a TF-IDF score associated with each word for each example/document
# Note: Depending on data set, execution may take a while
def tf_idf(list_of_words):
    # Get number of documents used (considered to appear as columns)
    list_dimensions = list_of_words.shape
    try:
        no_of_documents = list_dimensions[1]
    except IndexError:
        no_of_documents = 0

    # First step: Calculate frequency of words for each document
    # For first document
    no_words_expl_1 = list_of_words.iloc[:, 0].value_counts().sum()
    word_frequency = list_of_words.iloc[:, 0].value_counts() / no_words_expl_1

    # Other documents (columns are appended to word_frequency)
    for i in range(1, no_of_documents):  # start with second document
        current_document_no_of_words = list_of_words.iloc[:, 0].value_counts().sum()
        current_document_word_frequency = list_of_words.iloc[:, i].value_counts() / current_document_no_of_words

        word_frequency = pd.concat([word_frequency, current_document_word_frequency], axis=1, sort=True)

    overall_no_of_words = len(word_frequency.index)

    tf_idf_scores_result = word_frequency.copy()

    for k in range(0, no_of_documents):  # loop through columns (documents)
        for j in range(0, overall_no_of_words):  # loop through rows (words)
            index_name = tf_idf_scores_result.index[j]  # store name of current index so we know which word it is
            if math.isnan(tf_idf_scores_result.iat[j, k]):  # handle NaN issues
                tf_score = 0
            else:
                tf_score = tf_idf_scores_result.iat[j, k]
            idf_score = np.log(no_of_documents / no_documents_contain(list_of_words, index_name))
            tf_idf_scores_result.iloc[j, k] = tf_score * idf_score

    return tf_idf_scores_result


# Returns pandas.DataFrame with documents in columns and the text of the document as in the respective first row
def extract_text_from_directory(path_name):
    if path_name[-1:] == "/":   # logic for recognizing directories: has to end with "/"
        is_directory = True
    else:
        is_directory = False

    text = pd.DataFrame()     # construct empty DataFrame to be filled later
    if is_directory:    # assuming directory, write text for each document into one column
        i = 0
        for filename in os.listdir(path_name):
            extracted_text = codecs.open((path_name + filename), encoding="utf-8").read().lower()
            text.insert(i, filename, [extracted_text])  # write text into new column
            i += 1
    else:   # path_name is now interpreted as directly linking to a file
        extracted_text = codecs.open(path_name, encoding="utf-8").read().lower()  # read file, ensuring Int'l characters can be read
        text.insert(0, "Document1", extracted_text)

    return text


# Input: Text (unformatted)
# Output: list_of_words (pandas.DataFrame of dimensions no_words x 1) with words in rows
def extract_words_from_text(text):
    list_of_words = pd.DataFrame(re.findall(r"[\w']+", text))   # Using RegExp
    return list_of_words


def main():
    # Set categories: E. g. soccer, tennis, fighting (UFC, MMA), rugby, volleyball, hockey, basketball
    example_labels = np.array(["soccer", "fighting", "tennis", "soccer", "soccer",
                               "rugby", "rugby", "volleyball", "tennis", "tennis",
                               "tennis", "hockey", "hockey", "hockey", "soccer"
                                                                       "soccer", "soccer", "soccer", "basketball",
                               "fighting"])  # labels are ordered!

    path = "data/aa_bayes.tsv"  # Set path containing text documents as .txt files

    # words = extract_text_from_directory(path)  # Get words from text file into DataFrame

    # Extract information and save in DataFrame
    data_set = pd.read_csv(path, sep="\t")

    # Get words
    words = pd.DataFrame()

    for i in range(0, 15):          # keeping range low for the moment to keep runtime reasonable
        temp = extract_words_from_text(data_set.iat[i, 1])
        temp.columns = [i]
        words = pd.concat([words, temp], axis=1)

    # Get TF-IDF
    tf_idf_scores = tf_idf(words)

    a = 1
    # Select x highest scoring words for each category -> feature vectors

    # new training example
        # for each category: contains


    # ============ Old instructions ==============
    # Calculate feature vector frequency P(feature_i|class_j) = (no. of times feature_i appears in class_j) / (total count of features in class_j)
    # remember to use Laplace (or other) smoothing

    # Divide data set into training and validation set

    # Classify validation data and get results
    # Calculate feature vector frequency
    # Multiply P(feature_i|class_j) for all i, j, for each example
    # Choose class with highest likelihood for each example

    # Construct confusion matrix

    # Calculate accuracy, precision, true positives, false positives, F1-score

if __name__ ==  "__main__":
    main()