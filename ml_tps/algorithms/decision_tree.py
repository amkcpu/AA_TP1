import math
import numpy as np
import pandas as pd
from typing import List, Callable
from graphviz import Digraph

from ml_tps.utils.probability_utils import relative_frequency
from ml_tps.utils.data_processing import subdataframe, subdataframe_with_repeated
from ml_tps.utils.random_utils import random_string


# find leafs (= nodes that have no children) recursively
def findLeafs(node):
    if len(node.descendant_edges) == 0:
        return [node]
    else:
        return [item for edge in node.descendant_edges for item in findLeafs(edge.descendant)]


def findSubtreeFromNode(node):
    if len(node.descendant_edges) == 0:
        return [node]
    else:
        return [node] + [item for edge in node.descendant_edges for item in findSubtreeFromNode(edge.descendant)]

def findLeafParents(node):
    return [leaf.parent for leaf in findLeafs(node)]


def mostFrequentPrediction(branch_nodes_to_be_pruned: list):
    branch_values = list()

    for branch_node in branch_nodes_to_be_pruned:
        leafLabels = [leaf.label for leaf in findLeafs(branch_node)]
        occurences = pd.Series(leafLabels).value_counts()
        most_frequent_value = occurences.index[0]
        branch_values.append(most_frequent_value)

    return branch_values


class DecisionTree:

    class Node:
        def __init__(self, label: str, parent = None):
            self.label = label
            self.name = random_string(25)
            self.descendant_edges: List[DecisionTree.Edge] = []
            self.parent = parent

        def add_descendant_edge(self, value, descendant):
            self.descendant_edges.append(DecisionTree.Edge(value, descendant))

        def add_descendant_edges(self, descendant_tuples):
            for value, descendant in descendant_tuples:
                self.add_descendant_edge(value, descendant)

        def add_to_digraph(self, digraph):
            digraph.node(self.name, label=self.label)
            for edge in self.descendant_edges:
                edge.descendant.add_to_digraph(digraph)
                print(f"{self.label} -> {edge}")
                digraph.edge(tail_name=self.name, head_name=edge.descendant.name, label=str(edge.value))

        def remove_from_digraph(self, digraph):
            digraph.body = [entry for entry in digraph.body if not entry.__contains__(self.name) ]

        def classify(self, case: pd.Series):
            if len(self.descendant_edges) == 0:
                return self.label
            for edge in self.descendant_edges:
                if edge.value == case[self.label]:
                    return edge.descendant.classify(case)
            return "-1"

    class Edge:
        def __init__(self, value, descendant):
            self.value: str = value
            self.descendant: DecisionTree.Node = descendant

        def add_to_digraph(self, digraph, parent):
            self.descendant.add_to_digraph(digraph)
            print(f"{parent.label} -> {self}")
            digraph.edge(tail_name=parent.name, head_name=self.descendant.name, label=str(self.value))

    def __init__(self, dataset: pd.DataFrame, objective: str, gain_f: str = "shannon",
                 nodes: int = None, variables: int = None):
        if gain_f.lower() == "gini":
            gain_function = gini
        else:
            gain_function = shannon_entropy

        self.root = generate_subtree(dataset, objective, gain_f=gain_function, parent=None)
        self.digraph = self.generate_digraph()

    def classify(self, case: pd.Series):
        return self.root.classify(case)

    def plot(self, name_prefix: str = "", view=True):
        self.digraph.render(f'./out/DecisionTree_' + name_prefix + '_' + random_string(8) + '.png', view=view)

    def generate_digraph(self):
        dig = Digraph(format='png')
        self.root.add_to_digraph(dig)
        return dig

    def getPredictions(self, examples: pd.DataFrame, objective: str):
        predictions = pd.Series(np.array(np.zeros(len(examples.index), dtype=int)))

        i = 0
        for index, case in examples.drop(objective, axis=1, inplace=False).iterrows():
            predictions[i] = self.classify(case)
            i += 1

        return predictions


    # TODO
    def prune_tree(self, no_branches_to_be_pruned: int):
        pruned_tree = self
        leafParents = findLeafParents(pruned_tree.root)
        uniqueParents = list(set(leafParents))    # without repetitions

        branch_nodes_to_be_pruned = uniqueParents[:no_branches_to_be_pruned]

        # find most frequent class for one branch
        value_per_branch = mostFrequentPrediction(branch_nodes_to_be_pruned)

        # replace branch_node with one leaf having value of most frequent class
        i = 0
        for leafParent in branch_nodes_to_be_pruned:
            pruned_tree.replaceBranch(leafParent, leaf_value=value_per_branch[i])
            i += 1

        return pruned_tree

    def replaceBranch(self, branch_node_to_be_pruned, leaf_value):
        # Remove old nodes & edges
        for node in findSubtreeFromNode(branch_node_to_be_pruned):
            node.remove_from_digraph(self.digraph)

        newParent = branch_node_to_be_pruned.parent

        formerEdgeValue = ""
        i = 0
        for edge in newParent.descendant_edges:
            if edge.descendant is branch_node_to_be_pruned:
                formerEdgeValue = edge.value
                newParent.descendant_edges.pop(i)
                break
            i += 1

        node = DecisionTree.Node(label=leaf_value, parent=newParent)

        newParent.add_descendant_edge(value=formerEdgeValue, descendant=node)   # add to root
        DecisionTree.Edge(formerEdgeValue, node).add_to_digraph(self.digraph, newParent)    # add to digraph

    def no_of_nodes(self):
        return len(findSubtreeFromNode(self.root))


# pass node
def generate_subtree(dataset: pd.DataFrame, objective: str,
                     gain_f: Callable[[pd.DataFrame, str], float], parent: DecisionTree.Node):
    classes = list(dataset.keys())
    if len(classes) == 1:
        return DecisionTree.Node(str(dataset[objective].mode()[0]), parent)
    classes.remove(objective)

    if len(dataset[objective].unique()) == 1:           # special case: all examples have same value for objective
        return DecisionTree.Node(str(dataset[objective].unique()[0]), parent)
    # TODO: Add other special case (attributes empty): Return tree with one node returning most frequent value

    gain_list = np.array([gain(dataset=dataset, gain_f=gain_f, attribute=attr, objective=objective) for attr in classes])
    winner = classes[np.where(gain_list == np.amax(gain_list))[0][0]]
    values = dataset[winner].unique()
    node = DecisionTree.Node(winner, parent)
    subnodes = [generate_subtree(dataset=subdataframe(dataset, winner, v),
                                 gain_f=gain_f, objective=objective, parent=node) for v in values]
    node.add_descendant_edges(list(zip(values, subnodes)))
    return node


def shannon_entropy(dataset: pd.DataFrame, objective: str):
    # H(S) = -p(+) * log2(p(+)) - p(-) * log2(p(-))
    # if p+ = 0  then (-p(+) * log2(p(+))) is 0
    ## General
    # f(x = p(+)) = - x * log2(x) if x != 0 else 0
    # H(S) = sum( f(x) for x in values)
    f = lambda x: -x * math.log2(x) if x != 0 else 0
    frs = relative_frequency(dataset[objective])
    # As data set is an argument Sv is a subset of S
    return sum([f(x) for x in frs.values()])


def sv(dataset: pd.DataFrame, attribute: str, value) -> pd.DataFrame:
    return dataset[dataset[attribute] == value]


def gini(dataset: pd.DataFrame, objective: str):
    return 1 - sum(relative_frequency(dataset[objective]))


def gain(dataset: pd.DataFrame, gain_f: Callable[[pd.DataFrame, str], float], attribute: str, objective: str):
    svs = [sv(dataset, attribute, v) for v in dataset[attribute].unique()]
    general = gain_f(dataset, objective)
    return general - sum(len(_sv) / len(dataset) * gain_f(_sv, objective) for _sv in svs)
