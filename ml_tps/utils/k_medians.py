import pandas as pd
from ml_tps.utils.distance_utils import manhattan_distance
from ml_tps.utils.k_means import initialize_centroids, KMeans


def assign_centroids(X: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    """Assigns the nearest centroid to each training example."""
    X_assigned = X.copy()

    for idx, row in X.iterrows():
        distances = {i: manhattan_distance(row, r) for i, r in centroids.iterrows()}    # calc distance to each centroid
        closest_centroid = min(distances, key=distances.get)    # assign training example to closest centroid
        X_assigned.at[idx, "Centroid"] = closest_centroid

    return X_assigned


def move_centroids(X_assigned: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    """Moves the centroids to the median of their corresponding examples."""
    for idx, row in centroids.iterrows():
        assigned_rows = X_assigned[X_assigned["Centroid"] == idx]
        centroids.at[idx, :] = assigned_rows.drop("Centroid", axis=1).median(axis=0)

    return centroids


class KMedians:

    def __init__(self, initial_centroids: pd.DataFrame = None):
        self.centroids = initial_centroids

    def fit(self, X: pd.DataFrame, k: int, iters: int = 300, tol: float = 0.0001,
            initial_centroids: pd.DataFrame = None, plot_x_axis: str = None, plot_y_axis: str = None) -> None:
        """Clusters a given data set into k clusters using the K-Medians algorithm.

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
                print("Centroids stable. K-Medians finished.")
                break
            it += 1

        print("Finished after {} iterations.".format(it))
        print("Converged with error (cost) = {}.".format(error))

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Assigns each example to its closest centroid.

        :param X: DataFrame containing examples to be predicted.
        :return: Series containing the closest centroid for each example, identified by an int number.
        """
        return KMeans.predict(self=self, X=X)

    def cost(self, X_assigned: pd.DataFrame, costs_per_class: bool = False):
        """K-Medians cost function based on distances between data points and their assigned centroids.

        :param X_assigned: DataFrame containing examples in rows as well as the column "Centroid"
                            containing their assigned centroid.
        :param costs_per_class: If true, costs are returned as dict containing cost for each centroid.
        :return: Sum of distances of each data point to their assigned centroids
                or dict containing cost for each centroid separately.
        """
        return KMeans.cost(self=self, X_assigned=X_assigned, costs_per_class=costs_per_class)

    def plot(self, x_axis: str, y_axis: str, dataset: pd.DataFrame, plot_centroids: bool = True) -> None:
        '''Plots given data set along two specified dimensions, clustered around previously fit centroids (indicated by their color).

        :param x_axis: String specifying which column in the data set is to be used as x axis.
        :param y_axis: String specifying which column in the data set is to be used as y axis.
        :param dataset: Data set to be clustered and plotted.
        :param plot_centroids: Boolean specifying whether the previously fit centroids are to be plotted as well.
        '''
        KMeans.plot(self=self, x_axis=x_axis, y_axis=y_axis, dataset=dataset, plot_centroids=plot_centroids)
