import numpy as np
import pandas as pd
import math  # for NaN checking
import re  # for regexp splitting words
import codecs  # for unicode file reading
import os  # for looping over files in directory


def no_documents_contain(word_list: pd.DataFrame, word: str) -> int:
    """Returns the number of documents that a given word is found in from a given group of documents.

    :param word_list: A matrix where each column represents one document,
                                         and contains all words in said document.
    :param word: Word to be searched for.
    """
    counter = 0
    for col in word_list.columns:
        if word in word_list[col].values:
            counter += 1

    return counter


def tf_idf(list_of_words: pd.DataFrame) -> pd.DataFrame:
    """Implements the Term Frequency-Inverse Document Frequency (TF-IDF) algorithm, used to find important words in a group of documents.

    Note: Depending on data set size, execution can be very time-intensive/costly.

    :param list_of_words: A matrix where each column represents one document, and contains all words in said document.
    :returns: A matrix where each column represents one document, and contains the TF-IDF score of each word in said document.
    """
    no_of_documents = len(list_of_words.columns)

    # Calculate frequency of words for each document
    word_frequency = pd.DataFrame()

    for col in list_of_words.columns:
        current_document_word_frequency = list_of_words[col].value_counts(normalize=True)
        word_frequency = pd.concat([word_frequency, current_document_word_frequency], axis=1, sort=True)

    tf_idf_scores_result = word_frequency.copy()

    for col in word_frequency.columns:  # loop through columns (documents)
        for idx in word_frequency.index:  # loop through rows (words)
            if math.isnan(word_frequency.at[idx, col]):  # handle NaN issues
                tf_score = 0
            else:
                tf_score = word_frequency.at[idx, col]

            idf_score = np.log(no_of_documents / no_documents_contain(list_of_words, idx))
            tf_idf_scores_result.at[idx, col] = tf_score * idf_score

    return tf_idf_scores_result


def extract_words_from_text(text: str, prevent_uppercase_duplicates: bool = False):
    if prevent_uppercase_duplicates:
        text = text.lower()
    list_of_words = pd.DataFrame(re.findall(r"[\w']+", text))  # Using RegExp
    return list_of_words


def extract_text_from_directory(path_name: str) -> pd.DataFrame:
    """ Reads text from all documents found in a given directory into a DataFrame.

    :param path_name: Filepath to directory. Can also directly link to a single document.
    :return: Matrix with documents in columns and the text of the document in the first row.
    """
    if path_name[-1:] == "/":  # logic for recognizing directories: has to end with "/"
        is_directory = True
    else:
        is_directory = False

    text = pd.DataFrame()
    if is_directory:
        i = 0
        for filename in os.listdir(path_name):
            # read file, using UTF-8 encoding ensuring that international characters can be read
            extracted_text = codecs.open((path_name + filename), encoding="utf-8").read().lower()
            text.insert(i, filename, [extracted_text])
            i += 1
    else:  # path_name is now interpreted as directly linking to a single file
        extracted_text = codecs.open(path_name, encoding="utf-8").read().lower()
        text.insert(0, "Document1", extracted_text)

    return text
