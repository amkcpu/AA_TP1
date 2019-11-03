import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from ml_tps.utils.distance_utils import euclidean_distance


def get_closest_clusters(clusters: list, distance_method: str):
    total_distances = []
    for cluster in clusters:
        distances_per_cluster = {}
        for cl in clusters:
            distance = cluster.cluster_distance(other_cluster=cl, distance_method=distance_method)
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
    return [Cluster(index=idx, data=row) for idx, row in data.iterrows()]


class HierarchicalClustering:

    def __init__(self):
        self.Z = None
        self.clusters = None

    def fit(self, data: pd.DataFrame, distance_method: str, max_no_clusters: int) -> None:
        data.index = range(0, len(data))  # ensure that index increments from 0

        clusters = wrap_points_in_clusters(data)
        allow_clusters_setting = True
        Z = list()
        current_index = len(clusters)

        while len(clusters) > 1:
            if (len(clusters) <= max_no_clusters) and allow_clusters_setting:  # only set clusters once
                self.clusters = clusters.copy()
                allow_clusters_setting = False

            min_cluster1, min_cluster2, min_distance = get_closest_clusters(clusters, distance_method)
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

    def predict(self, data: pd.DataFrame, distance_method: str) -> pd.Series:
        predictions = []
        for idx, row in data.iterrows():
            example = Cluster(index=-1, data=row)
            distances = {cluster.index: example.cluster_distance(cluster, distance_method) for cluster in self.clusters}
            prediction = min(distances, key=distances.get)
            predictions.append(prediction)

        return pd.Series(predictions)

    def plot_dendrogram(self) -> None:
        if self.Z is None:
            raise ValueError("Model has not been fitted yet.")

        plt.figure()
        plt.xlabel("Data points")
        plt.ylabel("Distance")
        plt.title("Hierarchical clustering dendrogram", fontweight="bold")

        dn = dendrogram(self.Z)
        plt.show()

    def plot_clustering(self, x_axis: str, y_axis: str, data: pd.DataFrame, distance_method: str) -> None:
        predictions = self.predict(data=data, distance_method=distance_method)
        plt.scatter(data[x_axis], data[y_axis], c=predictions, s=50, cmap="Set3")
        plt.show()


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

    def construct_from_data(self, data: pd.DataFrame) -> None:
        data = pd.DataFrame(data).T     # as a Series, the mean() is not computed as required for this purpose
        self.members = data
        self.centroid = data.mean(axis=0)
        self.no_originals = len(data)

    def construct_from_clusters(self, cluster1, cluster2) -> None:
        self.members = pd.concat([cluster1.members, cluster2.members], ignore_index=True, axis=0)
        self.centroid = self.members.mean(axis=0)
        self.no_originals = cluster1.no_originals + cluster2.no_originals

    def cluster_distance(self, other_cluster, distance_method: str) -> float:
        methods = {"Centroid": ["cent", "centroid"],
                   "Max": ["max", "complete"],
                   "Min": ["min", "single"],
                   "Average": ["avg", "average"]}   # alternative names for consistency with SciPy implementation

        if distance_method in methods["Centroid"]:
            return euclidean_distance(self.centroid, other_cluster.centroid)
        elif distance_method in methods["Max"]:
            distances = []
            for idx, row in self.members.iterrows():
                distance_per_cluster = [euclidean_distance(row, r) for i, r in other_cluster.members.iterrows()]
                max_distance = max(distance_per_cluster)
                distances.append(max_distance)
            return max(distances)
        elif distance_method in methods["Min"]:
            distances = []
            for idx, row in self.members.iterrows():
                distance_per_cluster = [euclidean_distance(row, r) for i, r in other_cluster.members.iterrows()]
                min_distance = min(distance_per_cluster)
                distances.append(min_distance)
            return min(distances)
        elif distance_method in methods["Average"]:
            distances = []
            for idx, row in self.members.iterrows():
                distance_per_cluster = [euclidean_distance(row, r) for i, r in other_cluster.members.iterrows()]
                distances.extend(distance_per_cluster)
            return sum(distances) / len(distances)
        else:
            raise AttributeError('"{0}" is not a supported method for calculating cluster distance. '
                                 'The following dictionary lists the supported methods as keys, '
                                 'and the corresponding keywords as values: {1}.'.format(distance_method, methods))


data = pd.DataFrame([[1, 2], [1, 1], [5, 2], [3, 2], [1, 7], [2, 7], [2, 8], [5, 8], [4, 4], [4, 2]])
data2 = pd.DataFrame([[1, 1], [1, 2], [8, 8], [8, 9]])
plt.scatter(data[0], data[1])
plt.show()

method = "centroid"

cls = HierarchicalClustering()
cls.fit(data, distance_method=method, max_no_clusters=4)
cls.plot_dendrogram()
cls.plot_clustering(x_axis=0, y_axis=1, data=data, distance_method=method)

a = 1
