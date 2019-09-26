import pandas as pd
import numpy as np


# TODO revise if this is necessary with new confusion_matrix_diag() in place
def evalPreprocessing(predictions: pd.Series,
                      actual: pd.Series,
                      prediction_labels: list = None,
                      actual_labels: list = None):
    pred_saver = predictions
    ac_saver = actual
    if prediction_labels is not None:
        temp_pred = pd.Series()
        for index in predictions.index:
            if predictions[index] in prediction_labels:
                temp_pred = temp_pred.append(pd.Series([predictions[index]], index=[index]))
        predictions = temp_pred
        actual = actual[predictions.index]
    if actual_labels is not None:
        temp_actual = pd.Series()
        for index in actual.index:
            if actual[index] in actual_labels:
                temp_actual = temp_actual.append(pd.Series([predictions[index]], index=[index]))
        actual = temp_actual
        predictions = predictions[actual.index]

    if len(predictions) != len(actual):
        raise ValueError("Number of predictions does not equal number of validation examples (actual).")

    predicted_labels = predictions.value_counts().index
    actually_used_labels = actual.value_counts().index
    if len(predicted_labels) != len(actually_used_labels):
        raise ValueError("Number of predicted labels does not equal number of actual labels.")

    return predictions, actual


def getConfusionMatrix(predictions: pd.Series, actual: pd.Series):
    if len(predictions) != len(actual):
        raise ValueError("Mismatch between number of predictions and number of actual objective values.")

    predictions.index = actual.index        # so indices are equal
    return pd.crosstab(index=predictions,
                       columns=actual,
                       rownames=['Predicted'],
                       colnames=['Actual'],
                       dropna=False)


# Expects true objective values to be the columns
def getConfusionMatrixDiag(confusion_matrix: pd.DataFrame):
    cm_diag = {}
    for col in confusion_matrix.columns:
        try:
            cm_diag[col] = confusion_matrix.loc[col, col]
        except KeyError:
            continue

    return pd.Series(cm_diag)


def computeAccuracy(predictions: pd.Series,
                    actual: pd.Series):
    confusion_matrix = getConfusionMatrix(predictions, actual)
    nr_examples = confusion_matrix.sum().sum()
    nr_correctly_classified = getConfusionMatrixDiag(confusion_matrix).sum()

    return nr_correctly_classified / nr_examples


# TODO Fix TN and assure correctness in multi-class applications
# Returns parameters as vectors with entry for each objective class
def getEvaluationParameters(predictions: pd.Series,
                    actual: pd.Series,
                    prediction_labels: list = None,
                    actual_labels: list = None):
    predictions, actual = evalPreprocessing(predictions, actual,
                                            prediction_labels=prediction_labels,
                                            actual_labels=actual_labels)
    confusion_matrix = getConfusionMatrix(predictions, actual)
    confusion_matrix_diag = getConfusionMatrixDiag(confusion_matrix)

    FP = confusion_matrix.sum(axis=1) - confusion_matrix_diag
    FN = confusion_matrix.sum(axis=0) - confusion_matrix_diag
    TP = confusion_matrix_diag
    # TN for each class is the sum of all values of the confusion matrix excluding that class's row and column
    TN = confusion_matrix.sum().sum() - confusion_matrix.sum(axis=0) - confusion_matrix.sum(axis=1)
    # TN = confusion_matrix.sum() - (FP + FN + TP)

    '''
    if not as_vector_per_class:
        FP = FP.sum() / len(confusion_matrix_diag)
        FN = FN.sum() / len(confusion_matrix_diag)
        TP = TP.sum() / len(confusion_matrix_diag)
        TN = TN.sum() / len(confusion_matrix_diag)
    '''
    return FP, FN, TP, TN


def computeRecall(predictions: pd.Series,
                    actual: pd.Series,
                    prediction_labels: list = None,
                    actual_labels: list = None):
    predictions, actual = evalPreprocessing(predictions, actual,
                                            prediction_labels=prediction_labels,
                                            actual_labels=actual_labels)
    return computeTruePositiveRate(predictions, actual)


# TPR = sum(True positives)/sum(Condition positive)
def computeTruePositiveRate(predictions: pd.Series,
                    actual: pd.Series,
                    averaged: bool = True,
                    prediction_labels: list = None,
                    actual_labels: list = None):
    predictions, actual = evalPreprocessing(predictions, actual,
                                            prediction_labels=prediction_labels,
                                            actual_labels=actual_labels)
    confusion_matrix = getConfusionMatrix(predictions, actual)
    confusion_matrix_diag = getConfusionMatrixDiag(confusion_matrix)

    tpr = confusion_matrix_diag / confusion_matrix.sum(axis=0)

    if averaged:
        tpr = tpr.sum() / len(confusion_matrix_diag)

    return tpr


# FPR = sum(False positives)/sum(Condition negative)
# TODO Fix computation of False Positive Rate
def computeFalsePositiveRate(predictions: pd.Series, actual: pd.Series, averaged: bool = True,
                             prediction_labels: list = None,
                             actual_labels: list = None):
    predictions, actual = evalPreprocessing(predictions, actual,
                                            prediction_labels=prediction_labels,
                                            actual_labels=actual_labels)
    confusion_matrix = getConfusionMatrix(predictions, actual)
    confusion_matrix_diag = getConfusionMatrixDiag(confusion_matrix)

    false_positive_rate = 0
    for i in range(0, len(confusion_matrix_diag)):
        column_sum = confusion_matrix.iloc[:, i].sum()  # amount of examples actually in category
        classified_correctly = confusion_matrix_diag.iloc.iloc[i]
        classified_incorrectly = column_sum - classified_correctly

        false_positive_rate += classified_incorrectly / column_sum
    false_positive_rate /= len(confusion_matrix_diag)  # for average

    print("FPR (calc) = ", false_positive_rate)

    FP, FN, TP, TN = getEvaluationParameters(predictions, actual, False)

    condition_negative = FP + TN
    return FP / condition_negative


def computePrecision(predictions: pd.Series, actual: pd.Series, averaged: bool = True,
                     prediction_labels: list = None,
                     actual_labels: list = None):
    predictions, actual = evalPreprocessing(predictions, actual,
                                            prediction_labels=prediction_labels,
                                            actual_labels=actual_labels)
    confusion_matrix = getConfusionMatrix(predictions, actual)
    confusion_matrix_diag = getConfusionMatrixDiag(confusion_matrix)

    precision = confusion_matrix_diag / confusion_matrix.sum(axis=1)

    if averaged:
        precision = precision.sum() / len(confusion_matrix_diag)

    return precision


def f1_score(precision, recall):
    f1 = (2 * (precision * recall)) / (precision + recall)
    return f1
