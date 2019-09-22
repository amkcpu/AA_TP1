# Trabajo practico 2 - Ejercicio 2
from ml_tps.utils.dataframe_utils import divide_in_training_test_datasets
from ml_tps.utils.decision_tree_utils import DecisionTree
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FILEPATH = f"{dir_path}/../tp2/data/titanic.csv"
DEFAULT_OBJECTIVE = "Survived"
DEFAULT_TRAIN_PCTG = 0.6


def changesDaniel():
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


# Categorize age
def pour_titanic_dataset(dataset: pd.DataFrame):
    dataset = dataset[["Pclass", "Survived", "Sex", "Age"]]

    dataset["Age"] = dataset["Age"].fillna(-1)
    bins = (-2, 0, 5, 12, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Adult']
    categories = pd.cut(dataset["Age"], bins, labels=group_names)
    dataset["Age"] = categories

    return dataset


def main():
    objective = DEFAULT_OBJECTIVE
    training_percentage = DEFAULT_TRAIN_PCTG

    dataset = pd.read_csv(DEFAULT_FILEPATH, sep="\t")
    dataset = pour_titanic_dataset(dataset)

    # =========== a) Divide data set in two parts, training and evaluation set
    train, test = divide_in_training_test_datasets(dataset, train_pctg=training_percentage)

    # =========== b) Decision tree using Shannon entropy
    decision_tree = DecisionTree(train, objective=objective, gain_f="shannon")
    decision_tree.plot()

    # =========== c) Decision tree using Gini index
    decision_tree_gini = DecisionTree(train, objective=objective, gain_f="gini")
    decision_tree_gini.plot()

    # =========== d) Random forest for b) and c)


    # =========== e) Confusion matrix for b), c), d).1 and d).2
    # =========== f) Graph precision of decision tree vs. no. of nodes for each case

    a = 1

if __name__ == '__main__':
    changesDaniel()
    main()
