import pandas as pd
import numpy as np
from ml_tps.utils.distance_metric_utils import DistanceMetric
from ml_tps.utils.plotting_utils import plot_all_axes


def initialize_centroids(X: pd.DataFrame, k: int,
                         initial_centroids: pd.DataFrame = None,
                         seed: int = None,
                         current_centroids: pd.DataFrame = None) -> pd.DataFrame:
    """Randomly picks k examples from data set as centroids."""
    if initial_centroids is not None:
        if not len(initial_centroids.columns) == len(X.columns):
            raise ValueError("The centroids that were passed (initial_centroids) have a different dimension ({0}) "
                             "than the data set ({1}).".format(len(initial_centroids.columns), len(X.columns)))
        centroids = initial_centroids   # current centroids are overridden
    elif current_centroids is not None:
        if not len(current_centroids.columns) == len(X.columns):
            raise ValueError("Already assigned centroids have a different dimension ({0}) than the data set "
                             "that was passed ({1}).".format(len(current_centroids.columns), len(X.columns)))
        centroids = current_centroids
    else:
        np.random.seed(seed)
        indexes = np.arange(len(X))
        np.random.shuffle(indexes)
        indexes = list(indexes)
        centroids = pd.DataFrame([X.iloc[i] for i in indexes[:k]], index=range(0, k))

    return centroids


def assign_centroids(X: pd.DataFrame, centroids: pd.DataFrame, metric: str = "euclidean") -> pd.DataFrame:
    """Assigns the nearest centroid to each training example."""
    X_assigned = X.copy()

    distance = DistanceMetric(metric)
    for idx, row in X.iterrows():
        closest_centroid = distance.calculate_df(centroids, row).idxmin()
        X_assigned.at[idx, "Centroid"] = closest_centroid

    return X_assigned


def move_centroids(X_assigned: pd.DataFrame, centroids: pd.DataFrame, move_f_str: str = "mean") -> pd.DataFrame:
    """Moves the centroids to the mean of their corresponding examples."""
    for idx, row in centroids.iterrows():
        assigned_rows = X_assigned[X_assigned["Centroid"] == idx]
        if move_f_str.lower() == "mean":
            centroids.at[idx, :] = assigned_rows.drop("Centroid", axis=1).mean(axis=0)
        elif move_f_str.lower() == "median":
            centroids.at[idx, :] = assigned_rows.drop("Centroid", axis=1).median(axis=0)
        else:
            raise ValueError(f"Move method {move_f_str} not supported")

    return centroids


class KMeans:

    def __init__(self, initial_centroids: pd.DataFrame = None):
        self.centroids = initial_centroids
        self.move_f = "mean"
        self.metric = "euclidean"

    def fit(self, X: pd.DataFrame, k: int, iters: int = 300, tol: float = 0.0001,
            initial_centroids: pd.DataFrame = None, seed: int = None, plot_centroid_updates: bool = False) -> None:
        """Clusters a given data set into k clusters using the K-Means algorithm.

        :param X: Data set on which to perform clustering.
        :param k: Number of clusters to be found.
        :param iters: Maximum number of iterations before clustering is stopped.
        :param tol: Stop criteria for clustering. If the clustering error falls below the tolerance, clustering is stopped.
        :param initial_centroids: If so wished, the starting centroids can be passed.
        :param seed: Set seed for the random initialization of the centroids for reproducibility.
        :param plot_centroid_updates: If true, plots along each dimension are made on each iteration to be able to see the change of the centroids.
        """
        self.centroids = initialize_centroids(X, k, initial_centroids=initial_centroids, seed=seed, current_centroids=self.centroids)

        X_assigned = assign_centroids(X, self.centroids, self.metric)
        error = self.cost(X_assigned)
        it = 0
        while it < iters and error > tol:
            if plot_centroid_updates:
                self.plot(X=X, predictions=X_assigned["Centroid"], plot_centroids=True)

            centroids_prev = self.centroids.copy()
            self.centroids = move_centroids(X_assigned, self.centroids, self.move_f)
            X_assigned = assign_centroids(X, self.centroids, self.metric)

            error = self.cost(X_assigned)
            if self.centroids.equals(centroids_prev):    # break if centroids are stable
                print("Centroids stable. Clustering finished.")
                break
            it += 1

        print("Finished after {} iterations.".format(it))
        print("Converged with error (cost) = {}.".format(error))

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Assigns each example to its closest centroid.

        :param X: DataFrame containing examples to be predicted.
        :return: Series containing the closest centroid for each example, identified by an int number.
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet (centroids = None).")

        X_assigned = assign_centroids(X, self.centroids, self.metric)
        return X_assigned["Centroid"]

    def cost(self, X_assigned: pd.DataFrame, costs_per_class: bool = False, metric: str = "euclidean"):
        """K-Means cost function based on distances between data points and their assigned centroids.

        :param X_assigned: DataFrame containing examples in rows as well as the column "Centroid"
                            containing their assigned centroid.
        :param costs_per_class: If true, costs are returned as dict containing cost for each centroid.
        :param metric: Distance metric to be used for calculating the cost.
        :return: Sum of distances of each data point to their assigned centroids
                or dict containing cost for each centroid separately.
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet (centroids = None).")

        distance = DistanceMetric(metric)
        costs = dict()
        for idx, row in self.centroids.iterrows():
            assigned_rows = X_assigned[X_assigned["Centroid"] == idx].drop("Centroid", axis=1)
            costs[idx] = distance.calculate_df(assigned_rows, row).sum() / len(assigned_rows)

        if costs_per_class:
            return costs
        else:
            return sum(costs.values())

    def plot(self, X: pd.DataFrame, predictions: pd.Series, plot_centroids: bool = True) -> None:
        '''Plots given data set along all dimensions, clustered around previously fit centroids (indicated by their color).
        
        :param X: Data set to be clustered and plotted (does not include objective column).
        :param predictions: Predicted classes for each example as Series.
        :param plot_centroids: Boolean specifying whether the previously fit centroids are to be plotted as well.
        '''
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet (centroids = None).")

        if plot_centroids:
            plot_all_axes(data=X, predictions=predictions, additional_points=self.centroids)
        else:
            plot_all_axes(data=X, predictions=predictions)


class KMedians(KMeans):

    def __init__(self, initial_centroids: pd.DataFrame = None):
        super().__init__(initial_centroids)
        self.move_f = "median"
        self.metric = "manhattan"
