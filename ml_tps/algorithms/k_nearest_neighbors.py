import pandas as pd
import numpy as np
from ml_tps.utils.distance_metric_utils import DistanceMetric
from ml_tps.utils.plotting_utils import plot_all_axes


class KNN:

    def __init__(self):
        """Implements k-nearest neighbors algorithm."""
        self.X_train = None
        self.y_train = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Set data set for instance-based classification.

        :param X:     Training data wherein neighbors are searched. Attributes have to be numeric.
        :param y:     Column containing values for classification objective.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: pd.DataFrame, k: int, weighted: bool = False) -> pd.Series:
        """Predict class for given examples using the k-nearest neighbors algorithm.

        :param X: Examples to be predicted.
        :param k: Amount of neighbors considered in KNN.
        :param weighted: If true, the prediction weighs the distance of each neighbor to the given example.
        :return: Prediction for each example using given training data.
        :raises ValueError: When model has not been fitted yet.
        :raises ValueError: When example does not have the same amount of attributes as training data.
        """
        if (self.X_train is None) or (self.y_train is None):
            raise ValueError("Model has not been fitted yet.")
        if not len(X.columns) == len(self.X_train.columns):
            raise ValueError("Examples do not have the same amount of attributes as training data.")

        predictions = []
        for idx, row in X.iterrows():
            nearest_neighbors = get_nearest_neighbors(row, self.X_train, self.y_train, k)
            predicted_classes = prediction_per_class(nearest_neighbors, weighted)
            predicted = choose_predict_class(predicted_classes)
            predictions.append(predicted)

        return pd.Series(predictions)

    def plot(self, X: pd.DataFrame, predictions: pd.Series) -> None:
        """Plots given data set along all dimensions.

        :param X: Data set to be clustered and plotted (does not include objective column).
        :param predictions: Predicted classes for each example as Series.
        """
        plot_all_axes(data=X, predictions=predictions)


def get_nearest_neighbors(example: pd.Series, X_train: pd.DataFrame, y_train: pd.Series, k: int) -> pd.DataFrame:
    example.index = X_train.columns

    distance = DistanceMetric("euclidean")
    neighbors = distance.calculate_df(X_train, example)
    neighbors = pd.concat([neighbors, y_train], axis=1)
    neighbors.columns = ["Distance", "Class"]
    neighbors = neighbors.sort_values(by="Distance", ascending=True)

    return neighbors.head(k)  # grab k nearest neighbors


def prediction_per_class(nearest_neighbors: pd.DataFrame, weighted: bool) -> pd.Series:
    available_classes = nearest_neighbors["Class"].unique()  # which classes occur in nearest neighbors

    if weighted:  # If weighted, use 1/dist^2 as weight
        class_weight = lambda class_members: sum(1 / np.square(class_members["Distance"]))
    else:
        class_weight = lambda class_members: len(class_members)

    predicted_classes = []
    for this_class in available_classes:
        class_members = nearest_neighbors[nearest_neighbors["Class"] == this_class]
        predicted_classes.append(class_weight(class_members))

    return pd.Series(predicted_classes, index=available_classes).sort_values(ascending=False)


def choose_predict_class(predicted_classes: pd.Series) -> int:
    # Print error message if most frequently predicted classes have the same frequency
    if len(predicted_classes) > 1:
        if predicted_classes.iloc[0] == predicted_classes.iloc[1]:
            print("Prediction ambiguous: At least two classes appear equally often as nearest neighbors.")

    predicted = predicted_classes.head(1).index
    return predicted[0].astype(int)
