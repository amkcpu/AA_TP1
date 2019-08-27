import numpy as np
import pandas as pd
import math     # for NaN checking
import re       # for regexp splitting words
import codecs   # for unicode file reading
import os       # for looping over files in directory

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
    word_frequency = pd.DataFrame()

    for i in range(0, no_of_documents):
        current_document_no_of_words = list_of_words.iloc[:, i].value_counts().sum()
        current_document_word_frequency = list_of_words.iloc[:, i].value_counts() / current_document_no_of_words

        word_frequency = pd.concat([word_frequency, current_document_word_frequency], axis=1, sort=True)

    overall_no_of_words = len(word_frequency.index)

    tf_idf_scores_result = word_frequency.copy()

    for k in range(0, no_of_documents):  # loop through columns (documents)
        for j in range(0, overall_no_of_words):  # loop through rows (words)
            index_name = word_frequency.index[j]  # store name of current index so we know which word it is
            if math.isnan(word_frequency.iat[j, k]):  # handle NaN issues
                tf_score = 0
            else:
                tf_score = word_frequency.iat[j, k]
            idf_score = np.log(no_of_documents / no_documents_contain(list_of_words, index_name))
            tf_idf_scores_result.iloc[j, k] = tf_score * idf_score

    return tf_idf_scores_result

# Input: Text (unformatted)
# Output: list_of_words (pandas.DataFrame of dimensions no_words x 1) with words in rows
def extract_words_from_text(text, prevent_uppercase_duplicates=False):
    if prevent_uppercase_duplicates:
        text = text.lower()
    list_of_words = pd.DataFrame(re.findall(r"[\w']+", text))   # Using RegExp
    return list_of_words

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
