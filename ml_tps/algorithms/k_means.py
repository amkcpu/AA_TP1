import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ml_tps.utils.formulas import euclidean_distance


def initialize_centroids(X: pd.DataFrame, k: int,
                         initial_centroids: pd.DataFrame = None,
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
    else:
        indexes = np.arange(len(X))
        np.random.shuffle(indexes)
        indexes = list(indexes)
        centroids = pd.DataFrame([X.iloc[i] for i in indexes[:k]], index=range(1, k+1))

    return centroids


def assign_centroids(X: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    """Assigns the nearest centroid to each training example."""
    X_assigned = X.copy()

    for idx, row in X.iterrows():
        distances = {i: euclidean_distance(row, r) for i, r in centroids.iterrows()}    # calc distance to each centroid
        closest_centroid = min(distances, key=distances.get)    # assign training example to closest centroid
        X_assigned.at[idx, "Centroid"] = closest_centroid

    return X_assigned


def move_centroids(X_assigned: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    """Moves the centroids to the mean of their corresponding examples."""
    for idx, row in centroids.iterrows():
        assigned_rows = X_assigned[X_assigned["Centroid"] == idx]
        centroids.at[idx, :] = assigned_rows.drop("Centroid", axis=1).mean(axis=0)

    return centroids


class KMeans:

    def __init__(self, initial_centroids: pd.DataFrame = None):
        self.centroids = initial_centroids

    def fit(self, X: pd.DataFrame, k: int, iters: int = 300, tol: float = 0.0001,
            initial_centroids: pd.DataFrame = None, plot_x_axis: str = None, plot_y_axis: str = None) -> None:
        """Clusters a given data set into k clusters using the K-Means algorithm.

        :param X: Data set on which to perform clustering.
        :param k: Number of clusters to be found.
        :param iters: Maximum number of iterations before clustering is stopped.
        :param tol: Stop criteria for clustering. If the clustering error falls below the tolerance, clustering is stopped.
        :param initial_centroids: If so wished, the starting centroids can be passed.
        :param plot_x_axis: If this and plot_y_axis is passed, the data set is plotted on each iteration using these values as x and y axes.
        :param plot_y_axis: If this and plot_x_axis is passed, the data set is plotted on each iteration using these values as x and y axes.
        """
        self.centroids = initialize_centroids(X, k, initial_centroids=initial_centroids, current_centroids=self.centroids)

        X_assigned = assign_centroids(X, self.centroids)
        error = self.cost(X_assigned)
        it = 0
        while it < iters and error > tol:
            if (plot_x_axis is not None) and (plot_y_axis is not None):
                self.plot(x_axis=plot_x_axis, y_axis=plot_y_axis, dataset=X, plot_centroids=True)

            centroids_prev = self.centroids.copy()
            self.centroids = move_centroids(X_assigned, self.centroids)
            X_assigned = assign_centroids(X, self.centroids)

            error = self.cost(X_assigned)
            if self.centroids.equals(centroids_prev):    # break if centroids are stable
                print("Centroids stable. K-Means finished.")
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

        X_assigned = assign_centroids(X, self.centroids)
        return X_assigned["Centroid"]

    def cost(self, X_assigned: pd.DataFrame, costs_per_class: bool = False):
        """K-Means cost function based on distances between data points and their assigned centroids.

        :param X_assigned: DataFrame containing examples in rows as well as the column "Centroid"
                            containing their assigned centroid.
        :param costs_per_class: If true, costs are returned as dict containing cost for each centroid.
        :return: Sum of distances of each data point to their assigned centroids
                or dict containing cost for each centroid separately.
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet (centroids = None).")

        costs = dict()
        for idx, row in self.centroids.iterrows():
            assigned_rows = X_assigned[X_assigned["Centroid"] == idx].drop("Centroid", axis=1)
            costs[idx] = sum([euclidean_distance(row, r) for i, r in assigned_rows.iterrows()]) / len(assigned_rows)

        if costs_per_class:
            return costs
        else:
            return sum(costs.values())

    def plot(self, x_axis: str, y_axis: str, dataset: pd.DataFrame, plot_centroids: bool = True) -> None:
        '''Plots given data set along two specified dimensions, clustered around previously fit centroids (indicated by their color).

        :param x_axis: String specifying which column in the data set is to be used as x axis.
        :param y_axis: String specifying which column in the data set is to be used as y axis.
        :param dataset: Data set to be clustered and plotted.
        :param plot_centroids: Boolean specifying whether the previously fit centroids are to be plotted as well.
        '''
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet (centroids = None).")

        X_assigned = assign_centroids(X=dataset, centroids=self.centroids)
        plt.scatter(X_assigned[x_axis], X_assigned[y_axis], c=X_assigned["Centroid"], s=50, cmap="Set3")

        if plot_centroids:
            plt.scatter(self.centroids[x_axis], self.centroids[y_axis], c="black", marker="x", s=50)

        plt.show()
