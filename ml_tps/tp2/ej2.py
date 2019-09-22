# Trabajo practico 2 - Ejercicio 2
# Import data set
    # Note that titanic.csv is tab-separated (\t)
# a) Divide data set in two parts, training and evaluation set
# b) Decision tree using Shannon entropy
# c) Decision tree using Gini index
# d) Random forest for b) and c)
# e) Confusion matrix for b), c), d).1 and d).2
# f) Graph precision of decision tree vs. no. of nodes for each case

# Trabajo practico 2 - Ejercicio 2
# Import data set
    # Note that titanic.csv is tab-separated (\t)
# a) Divide data set in two parts, training and evaluation set
# b) Decision tree using Shannon entropy
# c) Decision tree using Gini index
# d) Random forest for b) and c)
# e) Confusion matrix for b), c), d).1 and d).2
# f) Graph precision of decision tree vs. no. of nodes for each case

from ml_tps.utils.decision_tree_utils import DecisionTree
import pandas as pd

DEFAULT_FILEPATH = "data/titanic.csv"
DEFAULT_OBJECTIVE = "Disfruta?"

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
    main()
