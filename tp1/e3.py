# Naive Bayes Text classifier
import numpy as np
import pandas as pd
import re  # for regexp splitting words
import codecs  # for unicode file reading
import os  # for looping over files in directory
import math  # for NaN checking


# Important for TF-IDF
def no_documents_contain(word_list, word):
    counter = 0
    no_examples, no_documents = word_list.shape

    for i in range(0, no_documents):
        if word in word_list.iloc[:, i].values:  # tests whether the word appears in a given column i
            counter += 1

    return counter


# Term Frequency-Inverse Document Frequency algorithm
# Explanation e.g. https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/
# Note: Depending on data set, execution may take a while
def tf_idf(list_of_words):
    # Get dimensions of the data set
    document_max_no_of_words, no_of_documents = list_of_words.shape

    # First step: Calculate frequency of words for each document
    # For first document
    no_words_expl_1 = list_of_words.iloc[:, 0].value_counts().sum()
    word_frequency = list_of_words.iloc[:, 0].value_counts() / no_words_expl_1

    # Other documents (columns are appended to word_frequency)
    for i in range(1, no_of_documents):  # start with second document
        temp2 = list_of_words.iloc[:, i].value_counts() / list_of_words.iloc[:,
                                                          0].value_counts().sum()  # frequency for one document
        word_frequency = pd.concat([word_frequency, temp2], axis=1, sort=True)

    # Useful constants
    overall_no_of_words, no_of_documents = word_frequency.shape

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


# Set categories: E. g. soccer, tennis, fighting (UFC, MMA), rugby, volleyball, hockey, basketball
example_labels = np.array(["soccer", "fighting", "tennis", "soccer", "soccer",
                           "rugby", "rugby", "volleyball", "tennis", "tennis",
                           "tennis", "hockey", "hockey", "hockey", "soccer"
                                                                   "soccer", "soccer", "soccer", "basketball",
                           "fighting"])  # labels are ordered!

# Get words from text file into DataFrame
path = "data/classificationDoc/sporty/"

for filename in os.listdir(path):
    text = codecs.open((path + filename),
                       encoding="utf-8").read().lower()  # read file, ensuring that Spanish characters can be read
    temp = pd.DataFrame(re.findall(r"[\w']+", text))  # extract words into DataFrame
    temp.columns = [filename]  # assign filename as column title

    try:  # error handling for first round
        words  # check if words is defined
    except NameError:
        words = temp  # if not (first round), assign temp and continue to next round
        continue

    words = pd.concat([words, temp], axis=1)  # else, concatenate existing words with new entry

# Get TF-IDF scores
tf_idf_scores = tf_idf(words)

a = 1
# Select x highest scoring words for each

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
