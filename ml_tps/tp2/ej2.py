# Trabajo practico 2 - Ejercicio 2
from ml_tps.utils.dataframe_utils import divide_in_training_test_datasets
from ml_tps.utils.decision_tree_utils import DecisionTree, RandomForest
import pandas as pd
import numpy as np
import os

from ml_tps.utils.evaluation_utils import getConfusionMatrix, computeAccuracy, evalPreprocessing

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
    view_trees = False

    dataset = pd.read_csv(DEFAULT_FILEPATH, sep="\t")
    dataset = pour_titanic_dataset(dataset)

    # =========== a) Divide data set in two parts, training and evaluation set
    train, test = divide_in_training_test_datasets(dataset, train_pctg=training_percentage)

    # =========== b) Decision tree using Shannon entropy
    decision_tree_shannon = DecisionTree(train, objective=objective, gain_f="shannon")
    decision_tree_shannon.plot(name_prefix="Shannon", view=True)

    # =========== c) Decision tree using Gini index
    decision_tree_gini = DecisionTree(train, objective=objective, gain_f="gini")
    decision_tree_gini.plot(name_prefix="Gini", view=view_trees)

    # =========== d) Random forest for b) and c)
    random_forest_shannon = RandomForest(train, objective=objective, gain_f="shannon")
    random_forest_shannon.plot(name_prefix="Shannon", view=view_trees)

    random_forest_gini = RandomForest(train, objective=objective, gain_f="gini")
    random_forest_gini.plot(name_prefix="Gini", view=view_trees)

    # =========== e) Confusion matrix for b), c), d).1 and d).2
    predictions_dt_shannon = decision_tree_shannon.getPredictions(test, objective)  # b)
    predictions_dt_gini = decision_tree_gini.getPredictions(test, objective)     # c)
    predictions_rf_shannon = random_forest_shannon.getPredictions(test, objective)  # d).1
    predictions_rf_gini = random_forest_gini.getPredictions(test, objective)     # d).2

    conf_matrix_dt_shannon = getConfusionMatrix(predictions_dt_shannon, test[objective])
    conf_matrix_dt_gini = getConfusionMatrix(predictions_dt_gini, test[objective])
    conf_matrix_rf_shannon = getConfusionMatrix(predictions_rf_shannon, test[objective])
    conf_matrix_rf_gini = getConfusionMatrix(predictions_rf_gini, test[objective])

    accuracy_dt_shannon = computeAccuracy(predictions_dt_shannon, test[objective], prediction_labels_to_drop=[-1])
    accuracy_dt_gini = computeAccuracy(predictions_dt_gini, test[objective], prediction_labels_to_drop=[-1])
    accuracy_rf_shannon = computeAccuracy(predictions_rf_shannon, test[objective], prediction_labels_to_drop=[-1])
    accuracy_rf_gini = computeAccuracy(predictions_rf_gini, test[objective], prediction_labels_to_drop=[-1])

    print("\n\n=======================================")
    print("Decision Tree - Shannon:")
    print("\tAccuracy = ", accuracy_dt_shannon)
    print(conf_matrix_dt_shannon, "\n")

    print("Decision Tree - Gini:")
    print("\tAccuracy = ", accuracy_dt_gini)
    print(conf_matrix_dt_gini, "\n")

    print("Random Forest - Shannon:")
    print("\tAccuracy = ", accuracy_rf_shannon)
    print(conf_matrix_rf_shannon, "\n")

    print("Random Forest - Gini:")
    print("\tAccuracy = ", accuracy_rf_gini)
    print(conf_matrix_rf_gini)

    # =========== f) Graph precision of decision tree vs. no. of nodes for each case
    # Decision tree pruning
    # Graph: Accuracy vs. no of nodes
        # For each case: b), c), d).1, d).2

    pruned_dt_shannon = decision_tree_shannon.prune_tree(2)
    '''
    for no_branches_to_be_pruned in range(1, 5):
        decision_tree_shannon_pruned = decision_tree_shannon.prune_tree(no_branches_to_be_pruned)
        decision_tree_gini_pruned = decision_tree_gini.prune_tree(no_branches_to_be_pruned)
        random_forest_shannon = random_forest_shannon.prune_forest(no_branches_to_be_pruned)
        random_forest_gini = random_forest_gini.prune_forest(no_branches_to_be_pruned)
    
        accuracy_dt_shannon_pruned = computeAccuracy(predictions_dt_shannon, test[objective], prediction_labels_to_drop=[-1])
        accuracy_dt_gini_pruned = computeAccuracy(predictions_dt_gini, test[objective], prediction_labels_to_drop=[-1])
        accuracy_rf_shannon_pruned = computeAccuracy(predictions_rf_shannon, test[objective], prediction_labels_to_drop=[-1])
        accuracy_rf_gini_pruned = computeAccuracy(predictions_rf_gini, test[objective], prediction_labels_to_drop=[-1])
    
        get no of nodes for each example
    
    collect no of nodes and accuracy
    
    plot graph, 4 lines (DT - Shannon; DT - Gini; RF - Shannon; RF - Gini)
    '''

    a = 1

if __name__ == '__main__':
    changesDaniel()
    main()
