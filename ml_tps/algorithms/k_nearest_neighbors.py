import pandas as pd
import numpy as np
from ml_tps.utils.data_processing import separate_dataset_objective_data
from ml_tps.utils.distance_metric_utils import DistanceMetric


def knn(example: pd.Series, training_set: pd.DataFrame, objective: str, k: int, weighted: bool):
    """Implements k-nearest neighbors algorithm.

    :param example:        Example to be predicted.
    :param training_set:   Training data wherein neighbors are searched. Attributes have to be numeric.
    :param objective:      Specifies which column in the training set is the objective column.
    :param k:              Amount of neighbors considered in KNN.
    :param weighted:       If true, the prediction weighs the distance of each neighbor to the given example.

    :returns: Prediction for given example using given training data.

    :raises ValueError: When example does not have the same amount of attributes as training data.
    """
    if not len(example) + 1 == len(training_set.columns):
        raise ValueError("Example does not have the same amount of attributes as training data.")

    nearest_neighbors = get_nearest_neighbors(example, training_set, objective, k)
    predicted_classes = prediction_per_class(nearest_neighbors, weighted)
    predicted = choose_predict_class(predicted_classes)

    return predicted


def get_nearest_neighbors(example: pd.Series, training_set: pd.DataFrame, objective: str, k: int):
    training_set_values, obj_values = separate_dataset_objective_data(dataset=training_set, objective=objective)
    example.index = training_set_values.columns

    distance = DistanceMetric("euclidean")
    neighbors = pd.Series([distance.calculate(row, example) for idx, row in training_set_values.iterrows()],
                          index=training_set_values.index)
    neighbors = pd.concat([neighbors, obj_values], axis=1)
    neighbors.columns = ["Distance", "Class"]
    neighbors = neighbors.sort_values(by="Distance", ascending=True)
    return neighbors.head(k)  # grab k nearest neighbors


def prediction_per_class(nearest_neighbors: pd.DataFrame, weighted: bool):
    available_classes = nearest_neighbors["Class"].unique()  # which classes occur in nearest neighbors
    predicted_classes = pd.Series(np.array(np.zeros(len(available_classes))), index=available_classes)

    for this_class in available_classes:
        class_members = nearest_neighbors[nearest_neighbors["Class"] == this_class]
        if weighted:  # If weighted, use 1/dist^2 as weight
            predicted_classes[this_class] = sum(1 / np.square(class_members["Distance"]))
        else:
            predicted_classes[this_class] = len(class_members["Class"])

    return predicted_classes.sort_values(ascending=False)


def choose_predict_class(predicted_classes: pd.Series):
    # Print error message if most frequently predicted classes have the same frequency
    if len(predicted_classes) > 1:
        if predicted_classes.iloc[0] == predicted_classes.iloc[1]:
            print("Prediction ambiguous: At least two classes appear equally often as nearest neighbors.")

    predicted = predicted_classes.head(1).index
    return predicted[0].astype(int)
