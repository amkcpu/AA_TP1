import pandas as pd
import numpy as np


def getConfusionMatrix(predictions: pd.Series, actual: pd.Series):
    if len(predictions) != len(actual):
        raise ValueError("Mismatch between number of predictions and number of actual objective values.")

    predictions.index = actual.index        # so indices are equal
    return pd.crosstab(actual,
                       predictions,
                       rownames=['Predicted'],
                       colnames=['Actual'],
                       dropna=False)


def getConfusionMatrixDiag(confusion_matrix: pd.DataFrame):
    return pd.Series(np.diag(confusion_matrix), index=confusion_matrix.index)


def computeAccuracy(predictions: pd.Series, actual: pd.Series):
    confusion_matrix = getConfusionMatrix(predictions, actual)
    confusion_matrix_diag = getConfusionMatrixDiag(confusion_matrix)
    return confusion_matrix_diag.sum() / confusion_matrix.sum().sum()


# Returns parameters as vectors with entry for each objective class
def getEvaluationParameters(predictions: pd.Series, actual: pd.Series, as_vector_per_class: bool):
    confusion_matrix = getConfusionMatrix(predictions, actual)
    confusion_matrix_diag = getConfusionMatrixDiag(confusion_matrix)

    FP = confusion_matrix.sum(axis=1) - confusion_matrix_diag
    FN = confusion_matrix.sum(axis=0) - confusion_matrix_diag
    TP = confusion_matrix_diag
    # TODO Fix TN
    #  TN for each class is the sum of all values of the confusion matrix excluding that class's row and column
    TN = confusion_matrix.sum(axis=0) - confusion_matrix.sum(axis=0) - confusion_matrix.sum(axis=1)
    # TN = confusion_matrix.sum() - (FP + FN + TP)

    if not as_vector_per_class:
        FP = FP.sum() / len(confusion_matrix_diag)
        FN = FN.sum() / len(confusion_matrix_diag)
        TP = TP.sum() / len(confusion_matrix_diag)
        TN = TN.sum() / len(confusion_matrix_diag)

    return FP, FN, TP, TN


def computeRecall(predictions: pd.Series, actual: pd.Series):
    return computeTruePositiveRate(predictions, actual)


# TPR = sum(True positives)/sum(Condition positive)
def computeTruePositiveRate(predictions: pd.Series, actual: pd.Series):
    FP, FN, TP, TN = getEvaluationParameters(predictions, actual, False)

    condition_positive = TP + FN
    return TP / condition_positive


# FPR = sum(False positives)/sum(Condition negative)
def computeFalsePositiveRate(predictions: pd.Series, actual: pd.Series):
    FP, FN, TP, TN = getEvaluationParameters(predictions, actual, False)

    condition_negative = FP + TN
    return FP / condition_negative


def computePrecision(predictions: pd.Series, actual: pd.Series):
    FP, FN, TP, TN = getEvaluationParameters(predictions, actual, False)

    predicted_positive = TP + FP
    return TP / predicted_positive


def f1_score(precision, recall):
    f1 = (2 * (precision * recall)) / (precision + recall)
    return f1
