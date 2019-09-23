import math
import numpy as np
import pandas as pd
from typing import List, Callable
from graphviz import Digraph

from ml_tps.utils.probability_utils import relative_frequency
from ml_tps.utils.dataframe_utils import subdataframe, subdataframe_with_repeated
from ml_tps.utils.random_utils import random_string


# TODO
# find leafs (= nodes that have no children)
    # then add their parent node to branch to be deleted
def findLeafs(decisionTree: DecisionTree):
    leafs = list()

    node = decisionTree.root
    for edge in node.descendant_edges:
        if edge.descendant.descendant_edges is None:
            leafs = leafs.append(edge.descendant)
            break
        else:
            node = edge.descendant

    return leafs


# TODO
def getLeafParents(leafs: list) -> pd.Series:
    return


def mostFrequentClass(branch_nodes_to_be_pruned: pd.Series):
    branch_values = list()

    for branch_node in branch_nodes_to_be_pruned:
        values_per_node = pd.Series()

        i = 0
        for edge in branch_node.descendant_edges:
            values_per_node[0] = edge.descendant.label
            i += 1

        values_per_node = values_per_node.value_counts()
        most_frequent_value = values_per_node.iloc[0]
        branch_values = branch_values.append(most_frequent_value)

    return branch_values


class DecisionTree:

    class Node:
        def __init__(self, label: str):
            self.label = label
            self.name = random_string(25)
            self.descendant_edges: List[DecisionTree.Edge] = []

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

    def __init__(self, dataset: pd.DataFrame, objective: str, gain_f: str = "shannon",
                 nodes: int = None, variables: int = None):
        if gain_f.lower() == "gini":
            gain_function = gini
        else:
            gain_function = shannon_entropy

        self.root = generate_subtree(dataset, objective, gain_f=gain_function)
        self.digraph = self.generate_digraph()

    def classify(self, case: pd.Series):
        return self.root.classify(case)

    def plot(self, name_prefix: str = "", view=True):
        self.digraph.render(f'./out/DecisionTree_' + name_prefix + '_' + random_string(8) + '.png', view=view)

    def generate_digraph(self):
        dig = Digraph(format='png')
        self.root.add_to_digraph(dig)
        return dig


    def prune_tree(self, no_branches_to_be_pruned: int):
        leafs = findLeafs(decisionTree=self, no_branches_to_be_pruned=no_branches_to_be_pruned)
        leafParents = getLeafParents(leafs)
        uniqueParents = leafParents.unique()    # without repetitions

        branch_nodes_to_be_pruned = uniqueParents[:no_branches_to_be_pruned]

        # find most frequent class for one branch
        value_per_branch = mostFrequentClass(branch_nodes_to_be_pruned)

        # replace all descendant edges of branch_node with one leaf having value of most frequent class
        for node in branch_nodes_to_be_pruned:
            prunedTree = deleteNodes(self, node.descendant_edges)   # delete nodes and associated edges

        return prunedTree

    def no_of_nodes(self):
        no_of_nodes = (self.digraph.body.__len__() + 1) / 2     # len consists of no. nodes and no. edges
                                                                # because of root node, there is one additional node
        return no_of_nodes


class RandomForest:

    def __init__(self, dataset: pd.DataFrame, objective: str, gain_f: str = "shannon",
                 trees: int = 5, trees_row_pctg: float = 0.9):
        self.dataset = dataset
        self.objective = objective
        self.gain_f = gain_f
        self.trees = [self._new_tree(trees_row_pctg) for _ in range(trees)]
        self.digraph = self.generate_digraph()

    def _new_tree(self, trees_row_pctg: float):
        df = subdataframe_with_repeated(self.dataset, int(len(self.dataset) * trees_row_pctg))
        return DecisionTree(df, "Survived")

    def gain_function(self, v):
        if self.gain_f.lower() == "gini" or (self.gain_f.lower() == "random" and np.random.random() < 0.5):
            return gini
        else:
            return shannon_entropy

    def classify(self, case):
        answers = list(filter(lambda x: x != "-1", [t.classify(case) for t in self.trees]))
        mode_array = pd.DataFrame(answers).mode()
        if len(mode_array) != 1:
            return "-1"
        return mode_array[0]

    def plot(self, name_prefix: str = "", view=True):
        self.digraph.render(f'./out/RandomForest_' + name_prefix + '_' + random_string(8) + '.png', view=view)

    def generate_digraph(self):
        dig = Digraph(format='png')
        for t in self.trees:
            t.root.add_to_digraph(dig)
        return dig


def generate_subtree(dataset: pd.DataFrame, objective: str, gain_f: Callable[[pd.DataFrame, str], float]):
    classes = list(dataset.keys())
    if len(classes) == 1:
        return DecisionTree.Node(str(dataset[objective].mode()[0]))
    classes.remove(objective)

    if len(dataset[objective].unique()) == 1:           # special case: all examples have same value for objective
        return DecisionTree.Node(str(dataset[objective].unique()[0]))
    # TODO: Add other special case (attributes empty): Return tree with one node returning most frequent value

    gain_list = np.array([gain(dataset=dataset, gain_f=gain_f, attribute=attr, objective=objective) for attr in classes])
    winner = classes[np.where(gain_list == np.amax(gain_list))[0][0]]
    values = dataset[winner].unique()
    subnodes = [generate_subtree(dataset=subdataframe(dataset, winner, v), gain_f=gain_f, objective=objective) for v in values]
    node = DecisionTree.Node(winner)
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
