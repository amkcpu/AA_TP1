import pandas as pd
import numpy as np
from graphviz import Digraph

from ml_tps.utils.random_utils import random_string
from ml_tps.algorithms.decision_tree import DecisionTree
from ml_tps.utils.data_processing import subdataframe_with_repeated
from ml_tps.utils.formulas import gini_index, shannon_entropy


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
            return gini_index
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

    def getPredictions(self, examples: pd.DataFrame, objective: str):
        predictions = pd.Series(np.array(np.zeros(len(examples.index), dtype=int)))

        i = 0
        for index, case in examples.drop(objective, axis=1, inplace=False).iterrows():
            predictions[i] = self.classify(case)
            i += 1

        return predictions

    # TODO
    def prune_forest(self, no_branches_to_be_pruned: int):
        i = 0
        for tree in self.trees:
            self.trees[i] = DecisionTree.prune_tree(tree, no_branches_to_be_pruned)
            i += 1

        # TODO update digraph
        return self
