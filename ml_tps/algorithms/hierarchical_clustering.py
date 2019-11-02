import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from ml_tps.utils.distance_utils import euclidean_distance


def get_closest_clusters(clusters: list, method: str):
    total_distances = []
    for cluster in clusters:
        distances_per_cluster = {}
        for cl in clusters:
            distance = cluster.cluster_distance(other_cluster=cl, method=method)
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
    return [Cluster(index=idx, data=pd.DataFrame(row).T) for idx, row in data.iterrows()]


class HierarchicalClustering:

    def __init__(self):
        self.Z = None

    def fit(self, data: pd.DataFrame, method: str) -> None:
        data.index = range(0, len(data))  # ensure that index increments from 0
        clusters = wrap_points_in_clusters(data)

        Z = list()
        current_index = len(clusters)
        while len(clusters) > 1:
            min_cluster1, min_cluster2, min_distance = get_closest_clusters(clusters, method)
            shared_no_of_originals = min_cluster1.no_originals + min_cluster2.no_originals

            Z.append([min_cluster1.index, min_cluster2.index, min_distance, shared_no_of_originals])

            # remove processed list elements
            clusters.remove(min_cluster1)
            clusters.remove(min_cluster2)
            clusters.append(Cluster(index=current_index, cluster1=min_cluster1, cluster2=min_cluster2))

            current_index += 1

        self.Z = pd.DataFrame(Z, columns=["Prev_Cluster 1", "Prev_Cluster 2", "Cluster Distance",
                                          "No. original data points in cluster"])

    def predict(self, examples: pd.DataFrame, no_clusters: int) -> pd.Series:
        # TODO
        pass
        # Error if no_clusters too big to classify all data points
        # Error if no_clusters < 2
        # Depending on no_clusters chosen, associate given input example with closest cluster

    def plot_dendrogram(self):
        if self.Z is None:
            raise ValueError("Model has not been fitted yet.")

        plt.figure()
        plt.xlabel("Data points")
        plt.ylabel("Distance")
        plt.title("Hierarchical clustering dendrogram", fontweight="bold")

        dn = dendrogram(self.Z)
        plt.show()

    def plot_clustering(self, x_axis: str, y_axis: str, no_clusters: int):
        # TODO
        # plot clustered examples
        # From KMeans:
        #  plt.scatter(X_assigned[x_axis], X_assigned[y_axis], c=X_assigned["Centroid"], s=50, cmap="Set3")
        pass


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

    def construct_from_data(self, data):
        self.members = data
        self.centroid = data.mean(axis=0)
        self.no_originals = len(data)

    def construct_from_clusters(self, cluster1, cluster2):
        self.members = pd.concat([cluster1.members, cluster2.members], ignore_index=True, axis=0)
        self.centroid = self.members.mean(axis=0)
        self.no_originals = cluster1.no_originals + cluster2.no_originals

    def cluster_distance(self, other_cluster, method: str):
        methods = {"Centroid": ["cent", "centroid"],
                   "Max": ["max", "complete"],
                   "Min": ["min", "single"],
                   "Average": ["avg", "average"]}   # alternative names for consistency with SciPy implementation

        if method in methods["Centroid"]:
            return euclidean_distance(self.centroid, other_cluster.centroid)
        elif method in methods["Max"]:
            distances = []
            for idx, row in self.members.iterrows():
                distance_per_cluster = [euclidean_distance(row, r) for i, r in other_cluster.members.iterrows()]
                max_distance = max(distance_per_cluster)
                distances.append(max_distance)
            return max(distances)
        elif method in methods["Min"]:
            distances = []
            for idx, row in self.members.iterrows():
                distance_per_cluster = [euclidean_distance(row, r) for i, r in other_cluster.members.iterrows()]
                min_distance = min(distance_per_cluster)
                distances.append(min_distance)
            return min(distances)
        elif method in methods["Average"]:
            distances = []
            for idx, row in self.members.iterrows():
                distance_per_cluster = [euclidean_distance(row, r) for i, r in other_cluster.members.iterrows()]
                distances.extend(distance_per_cluster)
            return sum(distances) / len(distances)
        else:
            raise AttributeError('"{0}" is not a supported method for calculating cluster distance. '
                                 'The following dictionary lists the supported methods as keys, '
                                 'and the corresponding keywords as values: {1}.'.format(method, methods))


Z_test = pd.DataFrame([[0, 1, 1.1, 2], [6, 2, 2.3, 3], [7, 3, 2.7, 4], [4, 5, 2.8, 2], [8, 9, 5.2, 7], [10, 11, 7.0, 9]],
                 columns=["Prev_Cluster 1", "Prev_Cluster 2", "Cluster Distance", "No. original examples contained"])
data = pd.DataFrame([[1, 2], [1, 1], [5, 2], [3, 2], [1, 7], [2, 7], [2, 8], [5, 8], [4, 4], [4, 2]],
                    index=[("Exp. " + str(i)) for i in range(0, 10)])
data2 = pd.DataFrame([[1, 1], [1, 2], [8, 8], [8, 9]])
plt.scatter(data[0], data[1])
plt.show()

method = "centroid"
plt.figure()
control_Z = linkage(data, method=method)
control_dn = dendrogram(control_Z)
plt.show()

cls = HierarchicalClustering()
cls.fit(data, method=method)
cls.plot_dendrogram()

a = 1
