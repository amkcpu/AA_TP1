# Trabajo practico 2 - Ejercicio 3

import pandas as pd
from ml_tps.utils.k_nearest_neighbors import knn
from ml_tps.utils.evaluation_utils import *
from ml_tps.utils.dataframe_utils import divide_in_training_test_datasets, \
    rewrite_positives_negatives, delete_non_numeric_columns, separate_dataset_objective_data
import datetime

DEFAULT_FILEPATH = "data/reviews_sentiment.csv"
DEFAULT_K = 5
TRAIN_PCTG = 0.6
DEFAULT_OBJECTIVE = "Star Rating"


def main():
    # Data import and cleaning
    dataset = pd.read_csv(DEFAULT_FILEPATH, sep=';')  # review_sentiments.csv is semicolon-separated (;)
    dataset = rewrite_positives_negatives(dataset)
    dataset = delete_non_numeric_columns(dataset)
    dataset["titleSentiment"] = dataset["titleSentiment"].fillna(dataset["textSentiment"])  # Handle NaN

    # ========== a) Mean no. of words of reviews valued with 1 star
    one_star_ratings = dataset[dataset["Star Rating"] == 1]
    one_star_review_mean_words = sum(one_star_ratings["wordcount"]) / len(one_star_ratings)

    # ========== b) Divide data set into two parts, training and evaluation set
    training_set, evaluation_set = divide_in_training_test_datasets(dataset=dataset, train_pctg=TRAIN_PCTG)
    evaluation_set_without_objective, orig_ratings = separate_dataset_objective_data(dataset=evaluation_set,
                                                                                     objective=DEFAULT_OBJECTIVE)

    # ========== c) Apply KNN and Weighted-distances KNN to predict review ratings (stars)
    time1 = datetime.datetime.now()
    predicted_ratings = evaluation_set_without_objective.apply(knn, axis=1,
                                                               args=(training_set, DEFAULT_OBJECTIVE, DEFAULT_K, False))
    time2 = datetime.datetime.now()
    print("Runtime Unweighted: ", divmod((time2 - time1).total_seconds(), 60), "\n")

    predicted_ratings_weighted = evaluation_set_without_objective.apply(knn, axis=1,
                                                                        args=(training_set, DEFAULT_OBJECTIVE, DEFAULT_K, True))
    print("Runtime Weighted: ", divmod((datetime.datetime.now() - time2).total_seconds(), 60), "\n")

    # ========== d) Calculate classifier precision and confusion matrix
    confusion_matrix = getConfusionMatrix(predicted_ratings, orig_ratings)
    accuracy = computeAccuracy(predicted_ratings, orig_ratings)
    true_positive_rate = computeTruePositiveRate(predicted_ratings, orig_ratings)
    precision = computePrecision(predicted_ratings, orig_ratings)
    recall = computeRecall(predicted_ratings, orig_ratings)
    f1 = f1_score(precision, recall)

    # KNN with weighted distances
    confusion_matrix_weighted = getConfusionMatrix(predicted_ratings_weighted, orig_ratings)
    accuracy_weighted = computeAccuracy(predicted_ratings_weighted, orig_ratings)
    true_positive_rate_weighted = computeTruePositiveRate(predicted_ratings_weighted, orig_ratings)
    precision_weighted = computePrecision(predicted_ratings_weighted, orig_ratings)
    recall_weighted = computeRecall(predicted_ratings_weighted, orig_ratings)
    f1_weighted = f1_score(precision_weighted, recall_weighted)

    # ============== Final printout ==============
    print("\n========== Ejercicio a) ==========")
    print("Mean no. of words of 1-star-reviews:", one_star_review_mean_words)

    print("\n\n========== Data info ==========")
    print("Data set dimensions: ", dataset.shape)
    print("Training set dimensions: ", training_set.shape)
    print("Evaluation set dimensions: ", evaluation_set.shape)
    print("Percentage of data set used for training: ", int(TRAIN_PCTG*100), "%")
    print("Classification objective: ", DEFAULT_OBJECTIVE)

    print("\n========== Evaluation metrics standard KNN ==========")
    print("Accuracy: ", accuracy, "\n")

    print("Confusion matrix:\n", confusion_matrix)

    print("\nTrue positive rate (TP) (= Recall): ", true_positive_rate)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)

    print("\n========== Evaluation metrics KNN with weighted distances ==========")
    print("Accuracy: ", accuracy_weighted, "\n")

    print("Confusion matrix:\n", confusion_matrix_weighted)

    print("\nTrue positive rate (TP) (= Recall): ", true_positive_rate_weighted)
    print("Precision: ", precision_weighted)
    print("Recall: ", recall_weighted)
    print("F1-score: ", f1_weighted)

    a = 1


if __name__ == '__main__':
    main()
