# Naive Bayes Text classifier
import numpy as np
import pandas as pd
import re  # for regexp splitting words
import codecs  # for unicode file reading
import os  # for looping over files in directory
import math  # for NaN checking
import datetime # measure runtime


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
        current_document_no_of_words = list_of_words.iloc[:, i].value_counts().sum()
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

    return tf_idf_scores_result, word_frequency


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
def extract_words_from_text(text, prevent_uppercase_duplicates=False):
    if prevent_uppercase_duplicates:
        text = text.lower()
    list_of_words = pd.DataFrame(re.findall(r"[\w']+", text))   # Using RegExp
    return list_of_words


def main():
    # Set categories: E. g. soccer, tennis, fighting (UFC, MMA), rugby, volleyball, hockey, basketball
    example_labels = np.array(["soccer", "fighting", "tennis", "soccer", "soccer",
                               "rugby", "rugby", "volleyball", "tennis", "tennis",
                               "tennis", "hockey", "hockey", "hockey", "soccer"
                                                                       "soccer", "soccer", "soccer", "basketball",
                               "fighting"])  # labels are ordered!

    # Adjust important variables
    path = "data/aa_bayes.tsv"  # Set path containing text documents as .txt files
    no_of_training_examples = 10    # data set size per category is around 3850 TODO do we want this to be a percentage?
    no_of_keywords = 20         # how many highest scoring words on TF-IDF are selected as features

    # Extract information and save in DataFrame
    data_set = pd.read_csv(path, sep="\t")

    # TODO find smarter way to separate given data sets
    data_set_deportes = data_set[data_set["categoria"]=="Deportes"]
    data_set_destacadas = data_set[data_set["categoria"]=="Destacadas"]
    data_set_economia = data_set[data_set["categoria"]=="Economia"]
    data_set_entretenimiento = data_set[data_set["categoria"]=="Entretenimiento"]
    data_set_internacional = data_set[data_set["categoria"]=="Internacional"]
    data_set_nacional = data_set[data_set["categoria"]=="Nacional"]
    data_set_salud = data_set[data_set["categoria"]=="Salud"]
    data_set_ciencia_tecnologia = data_set[data_set["categoria"]=="Ciencia y Tecnologia"]
    # Leaving out the massive, unspecific "Noticias destacadas" category

    categories = {"Deporte": data_set_deportes, "Destacadas": data_set_destacadas, "Economia": data_set_economia,
                  "Entretenimiento": data_set_entretenimiento, "Internacional": data_set_internacional,
                  "Nacional": data_set_nacional, "Salud": data_set_salud,
                  "CienciaTec": data_set_ciencia_tecnologia}

    # Get words
    # TODO Consider implementing Porter stemming to reduce redundancy
    #  http://www.3engine.net/wp/2015/02/stemming-con-python/
    words_then = datetime.datetime.now()    # for measuring runtime

    words = list()          # will contain words from all data subsets, each as one list element

    for subset_name, subset_data in categories.items():
        words_one_subset = pd.DataFrame()
        
        for i in range(0, no_of_training_examples):          # keeping range low for the moment to keep runtime reasonable
            words_one_title = extract_words_from_text(subset_data.iat[i, 1], True)
            words_one_title.columns = [(subset_name, i)]
            words_one_subset = pd.concat([words_one_subset, words_one_title], axis=1)

        words.append(words_one_subset)

    words_now = datetime.datetime.now()
    print("Runtime of word parsing: ", divmod((words_now - words_then).total_seconds(), 60), "\n")

    # Get TF-IDF
    then = datetime.datetime.now()  # perf measurement

    tf_idf_scores = list()
    word_frequencies = list()

    for i in range(0, len(categories)):
        tfidf_result, freq = tf_idf(words[i])     # word frequencies for Bayes classifier, contains NaN
        tf_idf_scores.append(tfidf_result)
        word_frequencies.append(freq)

    now = datetime.datetime.now()
    print("Runtime of TF-IDF: ", divmod((now - then).total_seconds(), 60))

    # Get sorted maximum TF-IDF scores for each category, with associated words as indices
    max_tfidf_scores = list()

    for i in range(0, len(categories)):
        temp = tf_idf_scores[i].max(axis=1).sort_values(ascending=False)
        max_tfidf_scores.append(temp)

    # Select x highest scoring words for each category -> feature vectors
    keywords = list()

    for i in range(0, len(categories)):
        temp = max_tfidf_scores[i].head(no_of_keywords)
        keywords.append(temp)

    # Retrieve frequency in respective category for each selected word -> parameter

    # Bayes classifier
        # examples should start from index no_of_training_examples
        # example.contains(list_of_keywords)
        # for keyword found: multiply keyword.frequency
        # at the end multiply with p(given_category)
        # find class with highest probability -> prediction

    # evaluation
        # confusion matrix
        # TP rate, FP rate
        # accuracy, precision, recall, F1-score



    a = 1
    # ============ Old instructions ==============
    # Calculate feature vector frequency P(feature_i|class_j) = (no. of times feature_i appears in class_j) / (total count of features in class_j)
    # remember to use Laplace (or other) smoothing

    # Divide data set into training and validation set

    # Classify validation data and get results
    # Calculate feature vector frequency
    # Multiply P(feature_i|class_j) for all i, j, for each example

if __name__ ==  "__main__":
    main()