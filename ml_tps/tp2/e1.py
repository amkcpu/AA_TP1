# Trabajo practico 2 - Ejercicio 1
# a) Implement decision tree with Shannon entropy
# b) Add additional training example and reconstruct decision tree

from ml_tps.utils.decision_tree_utils import DecisionTree
import pandas as pd

DEFAULT_FILEPATH = "data/deporte.csv"
DEFAULT_OBJECTIVE = "Disfruta?"


def changesDaniel():
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


def main():
    objective = DEFAULT_OBJECTIVE
    drop_extra_indexing_column = True
    extra_index_column_title = "Nro.Ejemplo"

    dataset = pd.read_csv(DEFAULT_FILEPATH)

    if drop_extra_indexing_column:
        del dataset[extra_index_column_title]   # Drop extra indexing column because of lacking value to classification

    # =========== EJ1 a) Create and generate Decision Tree ==========
    decision_tree = DecisionTree(dataset[:4], objective, "shannon")    # exclude example (5th example)
    decision_tree.plot(name_prefix="Shannon")

    # =========== EJ1 b) Add example ==========
    decision_tree_example_added = DecisionTree(dataset, objective, "shannon")
    decision_tree_example_added.plot(name_prefix="Shannon_Example_Added")

    a = 1

if __name__ == '__main__':
    changesDaniel()
    main()