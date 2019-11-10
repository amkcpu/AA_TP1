import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from ml_tps.utils.distance_metric_utils import DistanceMetric
from ml_tps.utils.plotting_utils import plot_all_axes


def get_closest_clusters(clusters: list, distance_method: str, distance_metric: str):
    """Searches for the two closest clusters in a list of clusters.

    :param clusters: Clusters to be searched in.
    :param distance_method: The distance method to be used for the distance computation.
    :param distance_metric: Distance metric to be used. Supports Euclidean ("euclidean", "l2") and Manhattan ("manhattan", "l1).
    :return: Returns the two closest clusters as objects and the calculated distance between them.
    """
    total_distances = []
    for cluster in clusters:
        distances_per_cluster = {}
        for cl in clusters:
            distance = cluster.cluster_distance(other_cluster=cl, distance_method=distance_method, distance_metric=distance_metric)
            distances_per_cluster[cl] = distance

        distances_per_cluster.pop(cluster)  # don't consider distance to itself
        min_cluster = min(distances_per_cluster, key=distances_per_cluster.get)
        min_distance = min(distances_per_cluster.values())

        total_distances.append([cluster, min_cluster, min_distance])

    dist = pd.DataFrame(total_distances)
    idxmin = dist[2].idxmin()
    min_cluster1 = dist.at[idxmin, 0]
    min_cluster2 = dist.at[idxmin, 1]
    min_distance = dist.at[idxmin, 2]

    return min_cluster1, min_cluster2, min_distance


def wrap_points_in_clusters(data: pd.DataFrame) -> list:
    data.index = range(0, len(data))  # ensure that index increments from 0
    return [Cluster(index=idx, data=row) for idx, row in data.iterrows()]


class HierarchicalClustering:

    def __init__(self):
        """Finds specified number of clusters in the passed data set using bottom-up hierarchical clustering.

        Uses two class variables:
            self.Z is oriented to the specification needed by SciPy's dendrogram method and is used precisely for plotting the dendrogram.

            self.clusters is a list containing the clusters as found by the model.
        """
        self.Z = None
        self.clusters = None

    def fit(self, X: pd.DataFrame, max_no_clusters: int, distance_method: str = "centroid",
            distance_metric: str = "euclidean", compute_full_tree: bool = True) -> None:
        """Fits the bottom-up hierarchical clustering model and sets the class variables self.Z and self.clusters.

        :param X: Data set to be clustered.
        :param max_no_clusters: Specifies the maximum number of clusters to be searched for by the algorithm.
        :param distance_method: Determines the method to be used for calculating the distance of the data points.
        :param distance_metric: Distance metric to be used. Supports Euclidean ("euclidean", "l2") and Manhattan ("manhattan", "l1).
        :param compute_full_tree: If set to False, the bottom-up clustering algorithm is interrupted as soon as
                                the specified number of clusters has been reached.
                                This may provide performance benefits if a dendrogram is not needed.
        """
        clusters = wrap_points_in_clusters(X)

        allow_clusters_setting = True
        Z = []
        current_index = len(clusters)
        while len(clusters) > 1:
            if (len(clusters) <= max_no_clusters) and allow_clusters_setting:  # only set clusters once
                self.clusters = clusters.copy()
                allow_clusters_setting = False
                if not compute_full_tree:
                    break

            min_cluster1, min_cluster2, min_distance = get_closest_clusters(clusters, distance_method, distance_metric)
            shared_no_of_originals = min_cluster1.no_originals + min_cluster2.no_originals

            Z.append([min_cluster1.index, min_cluster2.index, min_distance, shared_no_of_originals])

            # remove processed list elements
            clusters.remove(min_cluster1)
            clusters.remove(min_cluster2)
            clusters.append(Cluster(index=current_index, cluster1=min_cluster1, cluster2=min_cluster2))

            current_index += 1

        if max_no_clusters == 1:    # special case (is otherwise ignored by while loop)
            self.clusters = clusters.copy()

        self.Z = pd.DataFrame(Z, columns=["Prev_Cluster 1", "Prev_Cluster 2", "Cluster Distance",
                                          "No. original data points in cluster"])

    def predict(self, data: pd.DataFrame, distance_method: str, distance_metric: str) -> pd.Series:
        """Assigns each given example to the nearest cluster in the previously fitted model.

        :param data: Data to be assigned using the previously fitted clustering model.
        :param distance_method: Determines the method to be used for calculating the distance of the examples to the model's clusters.
        :param distance_metric: Distance metric to be used. Supports Euclidean ("euclidean", "l2") and Manhattan ("manhattan", "l1).
        :return: Series containing the index number of each example's nearest cluster.
        """
        if self.clusters is None:
            raise ValueError("Model has not been fitted yet.")

        predictions = []
        for idx, row in data.iterrows():
            example = Cluster(index=-1, data=row)
            distances = {cluster.index: example.cluster_distance(cluster, distance_method, distance_metric)
                         for cluster in self.clusters}
            prediction = min(distances, key=distances.get)
            predictions.append(prediction)

        return pd.Series(predictions)

    def plot_dendrogram(self) -> None:
        """Plot clustering's dendrogram using SciPy's dendrogram() method."""
        if (self.Z is None) and (self.clusters is None):
            raise ValueError("Model has not been fitted yet.")
        if (self.Z is None) and (self.clusters is not None):
            raise ValueError("Model has not been fully fitted. "
                             "Re-fit the model with the parameter compute_full_tree set to True.")

        plt.figure()
        plt.xlabel("Data points")
        plt.ylabel("Distance")
        plt.title("Hierarchical clustering dendrogram", fontweight="bold")

        dendrogram(self.Z)
        plt.show()

    def plot(self, X: pd.DataFrame, predictions: pd.Series) -> None:
        """Plots given data set along all dimensions.

        :param X: Data set to be clustered and plotted (does not include objective column).
        :param predictions: Predicted classes for each example as Series.
        """
        plot_all_axes(X, predictions)


class Cluster:

    def __init__(self, index: int, data: pd.DataFrame = None, cluster1=None, cluster2=None):
        self.index = index
        self.members = None
        self.centroid = None
        self.no_originals = None

        if data is not None:
            self.construct_from_data(data)
        elif (cluster1 is not None) and (cluster2 is not None):
            self.construct_from_clusters(cluster1, cluster2)
        else:
            raise ValueError("Cluster trying to be built with no data and less than 2 cluster")

    def construct_from_data(self, data: pd.DataFrame) -> None:
        data = pd.DataFrame(data).T     # as a Series, the mean() is not computed as required for this purpose
        self.members = data
        self.centroid = data.mean(axis=0)
        self.no_originals = len(data)

    def construct_from_clusters(self, cluster1, cluster2) -> None:
        self.members = pd.concat([cluster1.members, cluster2.members], ignore_index=True, axis=0)
        self.centroid = self.members.mean(axis=0)
        self.no_originals = cluster1.no_originals + cluster2.no_originals

    def cluster_distance(self, other_cluster, distance_method: str, distance_metric: str) -> float:
        """Calculates distance of self with the passed on cluster.

        :param other_cluster: Cluster to be compared to self.
        :param distance_method: Distance method to be used in comparison. Supports:
                                Distance between the cluster centroids/means ("centroid"),
                                maximum distance between all points in both clusters ("max"),
                                minimum distance between all points in both clusters ("min")
                                and average distance between all points in both clusters ("avg").
                                Multiple keywords per method are supported in order to provide compatibility
                                with SciPy's linkage() method.
        :param distance_metric: Distance metric to be used. Supports Euclidean ("euclidean", "l2") and Manhattan ("manhattan", "l1).
        :return: Distance between the two clusters as float.
        """
        methods = {"Centroid": ["cent", "centroid"],
                   "Max": ["max", "complete"],
                   "Min": ["min", "single"],
                   "Average": ["avg", "average"]}   # alternative names for consistency with SciPy implementation

        metric = DistanceMetric(distance_metric)

        # Select distance method
        if distance_method in methods["Centroid"]:
            return metric.calculate(self.centroid, other_cluster.centroid)
        elif distance_method in methods["Max"]:
            distances = []
            for idx, row in self.members.iterrows():
                distance_per_cluster = [metric.calculate(row, r) for i, r in other_cluster.members.iterrows()]
                max_distance = max(distance_per_cluster)
                distances.append(max_distance)
            return max(distances)
        elif distance_method in methods["Min"]:
            distances = []
            for idx, row in self.members.iterrows():
                distance_per_cluster = [metric.calculate(row, r) for i, r in other_cluster.members.iterrows()]
                min_distance = min(distance_per_cluster)
                distances.append(min_distance)
            return min(distances)
        elif distance_method in methods["Average"]:
            distances = []
            for idx, row in self.members.iterrows():
                distance_per_cluster = [metric.calculate(row, r) for i, r in other_cluster.members.iterrows()]
                distances.extend(distance_per_cluster)
            return sum(distances) / len(distances)
        else:
            raise AttributeError('"{0}" is not a supported method for calculating cluster distances. '
                                 'The following dictionary lists the supported methods as keys, '
                                 'and the corresponding keywords as values: {1}.'.format(distance_method, methods))
