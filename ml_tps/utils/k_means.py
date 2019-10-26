import pandas as pd
import numpy as np
from ml_tps.utils.distance_utils import euclidean_distance


def initialize_centroids(X: pd.DataFrame, k: int):
    """Randomly picks k examples from data set as centroids."""
    indexes = np.arange(len(X))
    np.random.shuffle(indexes)
    indexes = list(indexes)
    centroids = pd.DataFrame([X.iloc[i] for i in indexes[:k]])
    return centroids


class KMeans:

    def __init__(self, initial_centroids: pd.DataFrame = None):
        self.centroids = initial_centroids

    def fit(self, X: pd.DataFrame, k: int, iters: int = 1000, tol: float = 0.001):
        self.centroids = initialize_centroids(X, k)

        error = self.cost(X)
        it = 0
        while it < iters and error > tol:
            centroids_prev = self.centroids
            # TODO
            #  calc distance of each training example to each centroid
            #  assign each training example to its closest centroid
            #  move centroid -> calculate mean of all its assigned examples
            error = self.cost(X)
            it += 1
            if self.centroids.equals(centroids_prev):    # break if centroids are stable
                break

        print("Finished after {} iterations.".format(it))
        print("Converged with error (cost) = {}.".format(error))

    def predict(self, X: pd.DataFrame):
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet (centroids = None).")

        predictions = pd.Series(np.zeros(len(X)), index=X.index)
        for idx, row in X.iterrows():
            distances = pd.Series([euclidean_distance(centroid, row) for centroid in self.centroids],
                                  index=self.centroids.index)
            predictions[idx] = distances.sort_values(ascending=False).head(1).index

        return predictions

    def cost(self, X):
        n = len(X)
        # TODO


data = pd.DataFrame([[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]])
clf = KMeans()
a = 1
