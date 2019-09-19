# Trabajo practico 2 - Ejercicio 3

import pandas as pd
import numpy as np
import numbers
import math

from ml_tps.tp2.k_nearest_neighbors import knn
from ml_tps.utils.evaluation_utils import *
from ml_tps.utils.dataframe_utils import divide_in_training_test_datasets, rewritePositivesNegatives, deleteNonNumericColumns

DEFAULT_FILEPATH = "data/reviews_sentiment.csv"
DEFAULT_K = 5
DEFAULT_OBJECTIVE = "Star Rating"


def main():
    dataset = pd.read_csv(DEFAULT_FILEPATH, sep=';')  # review_sentiments.csv is semicolon-separated (;)
    dataset = rewritePositivesNegatives(dataset)
    dataset = deleteNonNumericColumns(dataset)

    # Data set specific changes
    dataset["titleSentiment"] = dataset["titleSentiment"].fillna(dataset["textSentiment"])

    # ========== a) Mean no. of words of reviews valued with 1 star
    one_star_ratings = dataset[dataset["Star Rating"] == 1]
    one_star_review_mean_words = sum(one_star_ratings["wordcount"]) / len(one_star_ratings)

    # ========== b) Divide data set into two parts, training and evaluation set
    training_set, evaluation_set = divide_in_training_test_datasets(dataset, 0.6)

    # ========== c) Apply KNN and Weighted-distances KNN to predict review ratings (stars)
    # DEFAULT_K = 5, DEFAULT_OBJECTIVE = "Star Rating"
    evaluation_set_without_objective = evaluation_set.copy()
    del evaluation_set_without_objective[DEFAULT_OBJECTIVE]

    predicted_ratings = pd.Series(np.array(np.zeros(len(evaluation_set.index), dtype=int)))
    i = 0
    for index, example in evaluation_set_without_objective.iterrows():
        predicted_ratings[i] = knn(example, training_set, DEFAULT_OBJECTIVE, DEFAULT_K, False)
        i += 1

    # ========== d) Calculate classifier precision and confusion matrix
    orig_ratings = evaluation_set[DEFAULT_OBJECTIVE]
    confusion_matrix = getConfusionMatrix(predicted_ratings, orig_ratings)
    accuracy = computeAccuracy(predicted_ratings, orig_ratings)
    true_positive_rate = computeTruePositiveRate(predicted_ratings, orig_ratings)
    false_positive_rate = computeFalsePositiveRate(predicted_ratings, orig_ratings)
    precision = computePrecision(predicted_ratings, orig_ratings)
    recall = true_positive_rate
    f1 = f1_score(precision, recall)

    # ============== Final printout ==============
    print("========== Data info ==========")
    print("Data set dimensions: ", dataset.shape)
    print("Training set dimensions: ", training_set.shape)
    print("Evaluation set dimensions: ", evaluation_set.shape)

    print("\n========== Ejercicio a) ==========")
    print("Mean no. of words of 1-star-reviews:", one_star_review_mean_words)

    print("\n\n========== Evaluation metrics KNN ==========")
    print("Accuracy: ", accuracy, "\n")

    print("Confusion matrix:\n", confusion_matrix)

    print("\nTrue positive rate (TP) (= Recall): ", true_positive_rate)
    print("False positive rate (FP): ", false_positive_rate)
    print("Precision: ", precision)
    print("F1-score: ", f1)

    a = 1


if __name__ == '__main__':
    main()
