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


def extract_words_from_text(text: str, prevent_uppercase_duplicates: bool = False) -> pd.Series:
    if prevent_uppercase_duplicates:
        text = text.lower()
    list_of_words = pd.Series(re.findall(r"[\w']+", text))  # Using RegExp
    return list_of_words


def extract_text_from_directory(path_name: str, encoding: str = "utf-8") -> pd.DataFrame:
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
            extracted_text = codecs.open((path_name + filename), encoding=encoding).read().lower()
            text.insert(i, filename, [extracted_text])
            i += 1
    else:  # path_name is now interpreted as directly linking to a single file
        extracted_text = codecs.open(path_name, encoding=encoding).read().lower()
        text.insert(0, "Document1", extracted_text)

    return text


def no_unique_words(text: str, normalize: bool, prevent_uppercase_duplicates: bool = True):
    """Number of unique words within a body of text, if required relative count.

    :param text: Text to be analized as String.
    :param normalize: If true, returns relative count (frequency).
    :param prevent_uppercase_duplicates: Sets text to lower case to prevent having differently-cased,
                        but identical words appearing as unique words.
    """
    extracted_words = extract_words_from_text(text, prevent_uppercase_duplicates)

    if normalize:
        return extracted_words.nunique() / len(extracted_words)
    else:
        return extracted_words.nunique()


def word_frequency(text: str, list_of_words: list, normalize: bool, average: bool = False, prevent_uppercase_duplicates: bool = True):
    """Return frequency (absolute or relative) of appearances of given words in given text.

    :param text: Text to be analyzed.
    :param list_of_words: List containing all words that are to be searched for in the text.
    :param normalize: If true, relative frequency is returned. Else, the absolute frequency.
    :param average: If group of words is passed, averages the frequency of all words to give the frequency of the word group.
    :param prevent_uppercase_duplicates: Sets text to lower case to prevent having differently-cased,
                        but identical words appearing as unique words.
    :returns: Series containing the passed words as index and the word frequencies as values.
    """
    counts = extract_words_from_text(text, prevent_uppercase_duplicates).value_counts(normalize=normalize)
    freq = pd.Series()
    for word in list_of_words:
        if prevent_uppercase_duplicates:
            word = word.lower()
        try:
            freq_this_word = counts[word]
        except KeyError:
            freq_this_word = 0
        freq[word] = freq_this_word

    if average:
        return freq.sum() / len(freq)

    return freq


def most_frequent_words(text: str, no_words: int, normalize: bool, prevent_uppercase_duplicates: bool = True) -> pd.Series:
    """Returns list of most frequent words.

    :param text: Text to be analyzed.
    :param no_words: How many words are to be returned.
    :param normalize: If true, relative frequency is returned. Else, the absolute frequency.
    :param prevent_uppercase_duplicates: Sets text to lower case to prevent having differently-cased,
                        but identical words appearing as unique words.
    :return: Series containing the most frequent words as index and the word frequencies as values.
    """
    words = extract_words_from_text(text, prevent_uppercase_duplicates)
    val_counts = words.value_counts(ascending=False, normalize=normalize)
    return val_counts.head(no_words)


def no_words_with_word_part(text: str, word_part: str, mode: str, normalize: bool, prevent_uppercase_duplicates: bool = True):
    """Searches for occurrences of given part of a word in the entire text.

    Can be used, for example, to search for English adverbs (ending in "-ly" - note that this approach is,
    of course, inherently error-prone).

    :param text: Text to be analyzed.
    :param word_part: String that is interpreted as part of a word and searched in the entire text.
    :param mode: Where in the words the word part is to be searched. Supports "beginning", "ending" and "containing".
    :param normalize: If true, relative frequency is returned. Else, the absolute frequency.
    :param prevent_uppercase_duplicates: Sets text to lower case to prevent having differently-cased,
                        but identical words appearing as unique words.
    :return: Absolute of relative frequency of words containing word_part.
    """
    extracted_words = extract_words_from_text(text, prevent_uppercase_duplicates)
    if prevent_uppercase_duplicates:
        word_part = word_part.lower()
        text = text.lower()

    if mode == "beginning":        # word begins with word_part
        matches = pd.Series(re.findall(r"\b" + word_part, text))
    elif mode == "ending":         # word ends with word_part
        matches = pd.Series(re.findall(word_part + r"\b", text))
    elif mode == "containing":     # word contains word_part somewhere
        matches = pd.Series(re.findall(r"[\w]*" + word_part + r"[\w]*", text))
    else:
        raise AttributeError('no_word_parts() only supports "beginning", "ending" and "containing" '
                             'as arguments for the parameter mode.')

    if normalize:
        return len(matches) / len(extracted_words)
    else:
        return len(matches)
