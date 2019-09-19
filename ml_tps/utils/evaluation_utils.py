import pandas as pd
import numpy as np


# F1-score
def f1_score(precision, recall):
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# TODO this actually contains an error occurring in small validation samples,
#  whereby some categories might not be predicted and the confusion matrix might
#  not by square. An error gets thrown when initializing confusion_matrix_diag, but
#  already the confusion_matrix itself is faulty.
#  Using a large sample, this problem is avoided.
def getConfusionMatrix(predictions: pd.Series, actual: pd.Series):
    if len(predictions) != len(actual):
        raise ValueError("Number of predictions and number of actual objective values mismatch.")

    actual.index = predictions.index        # so indices are equal
    return pd.crosstab(predictions,
                       actual,
                       rownames=['Actual'],
                       colnames=['Predicted'],
                       dropna=False)


def getConfusionMatrixDiag(confusion_matrix: pd.DataFrame):
    return pd.Series(np.diag(confusion_matrix), index=confusion_matrix.index)


def computeAccuracy(predictions: pd.Series, actual: pd.Series):
    confusion_matrix = getConfusionMatrix(predictions, actual)
    confusion_matrix_diag = getConfusionMatrixDiag(confusion_matrix)
    return confusion_matrix_diag.sum() / confusion_matrix.sum().sum()


def computeTruePositiveRate(predictions: pd.Series, actual: pd.Series):
    confusion_matrix = getConfusionMatrix(predictions, actual)
    confusion_matrix_diag = getConfusionMatrixDiag(confusion_matrix)

    true_positive_rate = 0
    for i in range(0, len(confusion_matrix_diag)):
        column_sum = confusion_matrix.iloc[:, i].sum()  # amount of examples actually in category
        classified_correctly = confusion_matrix_diag.iloc[i]

        true_positive_rate += classified_correctly / column_sum

    return true_positive_rate / len(confusion_matrix_diag)  # for average


# TODO Check if correct
def computeFalsePositiveRate(predictions: pd.Series, actual: pd.Series):
    confusion_matrix = getConfusionMatrix(predictions, actual)
    confusion_matrix_diag = getConfusionMatrixDiag(confusion_matrix)

    false_positive_rate = 0

    for i in range(0, len(confusion_matrix_diag)):
        column_sum = confusion_matrix.iloc[:, i].sum()  # amount of examples actually in category
        classified_correctly = confusion_matrix_diag.iloc[i]
        classified_incorrectly = column_sum - classified_correctly

        false_positive_rate += classified_incorrectly / column_sum

    return false_positive_rate / len(confusion_matrix_diag)  # for average


def computePrecision(predictions: pd.Series, actual: pd.Series):
    confusion_matrix = getConfusionMatrix(predictions, actual)
    confusion_matrix_diag = getConfusionMatrixDiag(confusion_matrix)

    precision = 0
    for i in range(0, len(confusion_matrix_diag)):
        precision += confusion_matrix_diag.iloc[i] / confusion_matrix.iloc[i, :].sum()

    return precision / len(confusion_matrix_diag)  # for average