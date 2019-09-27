# Trabajo Practico 3 - Ejercicio 2
import datetime
import pandas as pd
import numpy as np
import os
from sklearn import svm
from ml_tps.utils.evaluation_utils import getConfusionMatrix, computeAccuracy
from ml_tps.utils.dataframe_utils import divide_in_training_test_datasets, scale_dataset, seperateDatasetObjectiveData

dir_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FILEPATH = f"{dir_path}/../tp3/data/acath.xls"
DEFAULT_OBJECTIVE = "sigdz"
DEFAULT_TRAIN_PCTG = 0.6


def main():
    # a)  Divide dataset randomly into training and evaluation set
    dataset = pd.read_excel(DEFAULT_FILEPATH)
    dataset = dataset.dropna()      # TODO maybe deal with NaN otherwise?
    dataset_scaled = scale_dataset(dataset, objective=DEFAULT_OBJECTIVE, ignore_objective=True, scaling_type="minmax")

    train, test = divide_in_training_test_datasets(dataset_scaled, train_pctg=DEFAULT_TRAIN_PCTG)

    X_train, y_train = seperateDatasetObjectiveData(train, DEFAULT_OBJECTIVE)
    X_test, y_test = seperateDatasetObjectiveData(test, DEFAULT_OBJECTIVE)

    # b)  Classify categorical variable "sigdz" using default SVC SVM
    words_then = datetime.datetime.now()
    svm_values = pd.DataFrame(columns=["Kernel", "C value", "Accuracy"])
    c_value1 = 1
    kernel1 = "rbf"
    clf1 = svm.SVC(kernel=kernel1, gamma='scale', C=c_value1)      # using default parameters, written down for illustrative purposes
    clf1.fit(X_train, y_train)
    predictions_test = pd.Series(clf1.predict(X_test).T)
    confusion_matrix = getConfusionMatrix(predictions_test, y_test)
    accuracy1 = computeAccuracy(predictions_test, y_test)

    svm_values.loc[0] = [kernel1, c_value1, accuracy1]

    words_now = datetime.datetime.now()
    print("Runtime Default SVM fitting and testing: ", divmod((words_now - words_then).total_seconds(), 60), "\n")

    # c)  Evaluate different values for C and different nuclei to find better performing classifiers
    for kernel in ["rbf", "poly", "linear", "sigmoid"]:
        for c_value in np.logspace(-3, 2, 6):
            clf = svm.SVC(kernel=kernel, C=c_value, gamma="scale", cache_size=500)
            clf.fit(X_train, y_train)
            predictions = pd.Series(clf.predict(X_test).T)
            accuracy = computeAccuracy(predictions, y_test)

            svm_values.loc[svm_values.index.max() + 1] = [kernel, c_value, accuracy]

    time_now = datetime.datetime.now()
    print("\n\nRuntime parameter and kernel testing: ", divmod((time_now - words_now).total_seconds(), 60), "\n")

    # Choose SVM with highest accuracy
    winner = svm_values.sort_values(by="Accuracy", ascending=False).head(1)

    a = 1


if __name__ == '__main__':
    main()
