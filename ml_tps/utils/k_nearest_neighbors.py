import pandas as pd
import numpy as np
from ml_tps.utils.dataframe_utils import separate_dataset_objective_data
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
    training_set_values, obj_values = separate_dataset_objective_data(dataset=training_set, objective=objective)
    example.index = training_set_values.columns

    neighbors = pd.Series([euclidean_distance(row, example) for idx, row in training_set_values.iterrows()],
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
        if weighted:    # If weighted, use 1/dist^2 as weight
            predicted_classes[this_class] = sum(1 / np.square(class_members["Distance"]))
        else:
            predicted_classes[this_class] = len(class_members["Class"])

    return predicted_classes.sort_values(ascending=False)


def choose_predict_class(predicted_classes: pd.Series):
    # TODO: What if two classes have same frequency?
    predicted = predicted_classes.head(1).index
    return predicted[0].astype(int)
