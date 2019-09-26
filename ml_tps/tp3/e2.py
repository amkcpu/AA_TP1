# Trabajo Practico 3 - Ejercicio 2
import datetime

from ml_tps.utils.dataframe_utils import divide_in_training_test_datasets
import pandas as pd
import os
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from ml_tps.utils.evaluation_utils import getConfusionMatrix, computeAccuracy

dir_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FILEPATH = f"{dir_path}/../tp3/data/acath.xls"
DEFAULT_OBJECTIVE = "sigdz"
DEFAULT_TRAIN_PCTG = 0.6


def main():
    # a)  Divide dataset randomly into training and evaluation set
    dataset = pd.read_excel(DEFAULT_FILEPATH)
    dataset = dataset.dropna()      # TODO maybe deal with NaN otherwise?

    datasetX = dataset.loc[:, dataset.columns != DEFAULT_OBJECTIVE]
    scaler = MinMaxScaler()
    datasetX_scaled = pd.DataFrame(scaler.fit_transform(datasetX), index=datasetX.index, columns=datasetX.columns)
    dataset_scaled = pd.concat([datasetX_scaled, dataset[DEFAULT_OBJECTIVE]], axis=1)

    train, test = divide_in_training_test_datasets(dataset_scaled, train_pctg=DEFAULT_TRAIN_PCTG)

    X_train = train.loc[:, train.columns != DEFAULT_OBJECTIVE]
    y_train = train[DEFAULT_OBJECTIVE]

    X_test = test.loc[:, test.columns != DEFAULT_OBJECTIVE]
    y_test = test[DEFAULT_OBJECTIVE]

    words_then = datetime.datetime.now()
    # b)  Classify categorical variable "sigdz" using default SVC SVM
    classifiers = list()
    accuracies = list()
    classifiers.append(svm.SVC(kernel='rbf', gamma='auto', C=1))        # using default parameters, written down for illustrative purposes
    classifiers[0].fit(X_train, y_train)
    predictions_test = pd.Series(classifiers[0].predict(X_test).T)
    confusion_matrix = getConfusionMatrix(predictions_test, y_test)
    accuracies.append(computeAccuracy(predictions_test, y_test))

    words_now = datetime.datetime.now()
    print("Runtime SVM1: ", divmod((words_now - words_then).total_seconds(), 60), "\n")

    # c)  Evaluate different values for C and different nuclei to find better performing classifiers
    clf = svm.SVC(kernel='poly', degree=3, C=1, cache_size=500)
    clf.fit(X_train, y_train)
    predictions_test2 = pd.Series(clf.predict(X_test).T)
    confusion_matrix2 = getConfusionMatrix(predictions_test2, y_test)

    time_now = datetime.datetime.now()
    print("Runtime SVM2: ", divmod((time_now - words_now).total_seconds(), 60), "\n")

    a = 1


if __name__ == '__main__':
    main()
