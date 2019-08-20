# Naive Bayes Text classifier
import numpy as np
import pandas as pd
import re               # for regexp splitting words
import codecs           # for unicode file reading
import os               # for looping over files in directory

# Set categories: E. g. soccer, tennis, fighting (UFC, MMA), rugby, volleyball, hockey, basketball
example_labels = np.array(["soccer", "fighting", "tennis", "soccer", "soccer",
                           "rugby", "rugby", "volleyball", "tennis", "tennis",
                           "tennis", "hockey", "hockey", "hockey", "soccer"
                           "soccer", "soccer", "soccer", "basketball", "fighting"])     # labels are ordered!

# Get words from text file into DataFrame
path = "data/classificationDoc/sporty/"

for filename in os.listdir(path):
    text = codecs.open((path + filename), encoding="utf-8").read().lower()  # read file, ensuring that Spanish characters can be read
    temp = pd.DataFrame(re.findall(r"[\w']+", text))            # extract words into DataFrame
    temp.columns = [filename]                                   # assign filename as column title

    try:                                                        # error handling for first round
        words                                                   # check if words is defined
    except NameError:
        words = temp                                            # if not (first round), assign temp and continue to next round
        continue

    words = pd.concat([words, temp], axis=1)                    # else, concatenate existing words with new entry

# Select feature vectors (appearance of specific words) or TF-IDF for generalization
# (see https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/)
no_of_words, no_of_examples = words.shape
for i in range(0, no_of_examples):
    counts_in_rows = words.iloc[:, i].value_counts()

# Calculate feature vector frequency P(feature_i|class_j) = (no. of times feature_i appears in class_j) / (total count of features in class_j)
    # remember to use Laplace (or other) smoothing

# Divide data set into training and validation set

# Classify validation data and get results
    # Calculate feature vectore frequency
    # Multiply P(feature_i|class_j) for all i, j, for each example
    # Choose class with highest likelihood for each example

# Construct confusion matrix

# Calculate accuracy, precision, true positives, false positives, F1-score
