import pandas as pd
import numpy as np
from ml_tps.utils.dataframe_utils import drop_objective_column
from ml_tps.utils.distance_utils import euclidean_distance


# Input: DataFrame with numeric attributes and label in last column
def knn(example: pd.Series, training_set: pd.DataFrame, objective: str, k: int, weighted: bool):
    if not len(example) + 1 == len(training_set.columns):
        raise ValueError("Example does not have the same amount of attributes as training data.")

    nearest_neighbors = get_nearest_neighbors(example, training_set, objective, k)
    predicted_classes = prediction_per_class(nearest_neighbors, weighted)
    predicted = choose_predict_class(predicted_classes)

    return predicted


def get_nearest_neighbors(example: pd.Series, training_set: pd.DataFrame, objective: str, k: int):
    training_set_values = drop_objective_column(training_set, objective)

    # Transpose training set & set example columns to have correct subtraction
    example.index = training_set_values.columns
    training_set_values = training_set_values.transpose()

    # DataFrame to store calculated distances and objective value
    distances_and_objective = pd.DataFrame(np.array(np.zeros([len(training_set_values.columns), 2])),
                                           index=training_set_values.columns,
                                           columns=["Distance", "Objective Value"])

    # Calc distance and write objective
    for element in training_set_values:
        distances_and_objective["Distance"][element] = euclidean_distance(example, training_set_values[element])
        distances_and_objective["Objective Value"][element] = training_set[objective].loc[element]

    distances_and_objective = distances_and_objective.sort_values(by="Distance", ascending=True)
    return distances_and_objective.head(k)  # grab k nearest neighbors


def prediction_per_class(nearest_neighbors: pd.DataFrame, weighted: bool):
    available_classes = nearest_neighbors["Objective Value"].unique()  # which classes occur in nearest neighbors
    predicted_classes = pd.Series(np.array(np.zeros(len(available_classes))), index=available_classes)

    for this_class in available_classes:
        class_members = nearest_neighbors[nearest_neighbors["Objective Value"] == this_class]
        if weighted:
            predicted_classes[this_class] = sum(
                1 / np.square(class_members["Distance"]))  # If weighted, use 1/dist^2 as weight
        else:
            predicted_classes[this_class] = len(class_members["Objective Value"])

    return predicted_classes.sort_values(ascending=False)


def choose_predict_class(predicted_classes: pd.Series):
    # TODO: What if two classes have same frequency?
    predicted = predicted_classes.head(1).index
    return predicted[0].astype(int)
