import math
import numpy as np
import pandas as pd
from typing import List, Callable
from graphviz import Digraph

from ml_tps.utils.probability_utils import relative_frequency
from ml_tps.utils.dataframe_utils import subdataframe

from ml_tps.utils.random_utils import random_string


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

    def plot(self, save=False):
        self.digraph.render(f'./out/{random_string(8)}.png', view=not save)

    def generate_digraph(self):
        dig = Digraph(format='png')
        self.root.add_to_digraph(dig)
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
