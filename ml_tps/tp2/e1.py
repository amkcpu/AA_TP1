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
    drop_nro_example_column = True
    example_column_title = "Nro.Ejemplo"

    dataset = pd.read_csv(DEFAULT_FILEPATH)

    if drop_nro_example_column:
        del dataset[example_column_title]   # Drop nro. example because of lacking value to classification

    # =========== EJ1 a) Create and generate Decision Tree ==========
    decision_tree = DecisionTree(dataset[:4], objective, "shannon")    # exclude example (5th example)
    decision_tree.plot()

    # =========== EJ1 b) Add example ==========
    decision_tree_example_added = DecisionTree(dataset, objective, "shannon")
    decision_tree_example_added.plot()

    a = 1

if __name__ == '__main__':
    changesDaniel()
    main()