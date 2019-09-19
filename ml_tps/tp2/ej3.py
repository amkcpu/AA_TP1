# Trabajo practico 2 - Ejercicio 3

import pandas as pd
import numpy as np
import numbers
import math

from ml_tps.utils.evaluation_utils import f1_score
from ml_tps.utils.dataframe_utils import divide_in_training_test_datasets

DEFAULT_FILEPATH = "data/reviews_sentiment.csv"
DEFAULT_K = 5

# Rewrite negatives and positives to 0 and 1
def rewritePositivesNegatives(dataset: pd.DataFrame):
    for i in dataset.columns:
        dataset[i] = dataset[i].apply(lambda x: 1 if x == "positive" else 0 if x == "negative" else x)
    return dataset


# Based on check whether first element in column is Number or not
def deleteNonNumericColumns(dataset: pd.DataFrame):
    for i in dataset.columns:
        first_element_in_column = dataset[i].iloc[0]
        if not isinstance(first_element_in_column, numbers.Number):
            del dataset[i]
    return dataset


# Assumes that only numeric values are passed
def euclideanDistance(x1: pd.Series, x2: pd.Series):
    if len(x1) != len(x2):
        raise ArithmeticError("Vectors must have same length.")
    return sum(np.sqrt(np.square(x1 - x2)))


# Input: DataFrame with numeric attributes and label in last column
def knn(example: pd.Series, training_set: pd.DataFrame, objective: str, k: int, weighted: bool):
    # find k nearest neighbors
    if not len(example) + 1 == len(training_set.columns):
        raise ValueError("Example does not have the same amount of attributes as training data.")

    # Drop objective column
    training_set_values = training_set.copy()
    del training_set_values[objective]

    # Set example columns to have correct subtraction
    example.index = training_set_values.columns
    training_set_values = training_set_values.transpose()

    distances_and_objective = pd.DataFrame(np.array(np.zeros([len(training_set_values.columns), 2])))
    distances_and_objective.index = training_set_values.columns
    distances_and_objective.columns = ["Distance", "Objective Value"]

    # Calc distance and write objective
    for element in training_set_values:
        distances_and_objective["Distance"][element] = euclideanDistance(example, training_set_values[element])
        distances_and_objective["Objective Value"][element] = training_set[objective].loc[element]

    distances_and_objective = distances_and_objective.sort_values(by="Distance", ascending=True)
    nearest_neighbors = distances_and_objective.head(k)

    available_classes = nearest_neighbors["Objective Value"].unique()   # which classes are actually used
    predicted_classes = pd.Series(np.array(np.zeros(len(available_classes))), index=available_classes)

    weight = 1
    for value in available_classes:
        temp = nearest_neighbors[nearest_neighbors["Objective Value"] == value]
        if weighted:
            predicted_classes[value] = sum(1 / np.square(temp["Distance"])) # If weighted, use 1/dist(x_q,x_i)^2 as weight
        else:
            predicted_classes[value] = len(temp["Objective Value"])

    predicted_classes = predicted_classes.sort_values(ascending=False)

    # If two classes have same frequency
    if predicted_classes.iloc[0] != predicted_classes.iloc[1]:
        predicted = predicted_classes.head(1).index
    else:
        raise ValueError("Classification inconclusive. More than one class is equally probable. Try adjusting the parameter k.")

    return predicted[0].astype(int)


def main():
    dataset = pd.read_csv(DEFAULT_FILEPATH, sep=';')  # review_sentiments.csv is semicolon-separated (;)
    dataset = rewritePositivesNegatives(dataset)
    dataset = deleteNonNumericColumns(dataset)

    # Data set specific changes
    dataset["titleSentiment"] = dataset["titleSentiment"].fillna(dataset["textSentiment"])
    dataset["titleSentiment"] = dataset["titleSentiment"].astype(int)

    # ========== a) Mean no. of words of reviews valued with 1 star
    one_star_ratings = dataset[dataset["Star Rating"] == 1]
    one_star_review_mean_words = sum(one_star_ratings["wordcount"]) / len(one_star_ratings)

    # ========== b) Divide data set into two parts, training and evaluation set
    training_set, evaluation_set = divide_in_training_test_datasets(dataset, 0.6)
    example = pd.Series([20, 0, 0, -0.5])
    predicted_rating = knn(example, training_set, "Star Rating", DEFAULT_K, True)

    # ========== c) Apply KNN and Weighted-distances KNN to predict review ratings (stars)
    # DEFAULT_K = 5

    # ========== d) Calculate classifier precision and confusion matrix
    # pd.crosstab(validation_example_predictions, validation_examples_actual, rownames=['Actual'], colnames=['Predicted'])

    # ============== Final printout ==============
    print("========== Data info ==========")
    print("Data set dimensions: ", dataset.shape)
    print("Training set dimensions: ", training_set.shape)
    print("Evaluation set dimensions: ", evaluation_set.shape)

    print("\n========== Ejercicio a) ==========")
    print("Mean no. of words of 1-star-reviews:", one_star_review_mean_words)

    print("\n========== Evaluation metrics KNN ==========")
    '''
    print("Accuracy: ", accuracy, "\n")
    print("Confusion matrix:", confusion_matrix)
    print("\nTrue positive rate (TP): ", true_positive_rate)
    print("False positive rate (FP): ", false_positive_rate)
    print("Precision: ", precision)
    print("Recall (= true positive rate): ", recall)
    print("F1-score: ", f1)
    '''

    a = 1


if __name__ == '__main__':
    main()
