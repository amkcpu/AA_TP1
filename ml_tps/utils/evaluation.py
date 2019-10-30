import pandas as pd


def evalPreprocessing(predictions: pd.Series, actual: pd.Series):
    if len(predictions) != len(actual):
        raise ValueError("Number of predictions does not equal number of validation examples (actual).")

    predictions.index = actual.index

    return predictions, actual


def getConfMatrix_Diag(predictions: pd.Series, actual: pd.Series):
    confusion_matrix = getConfusionMatrix(predictions, actual)
    confusion_matrix_diag = getConfusionMatrixDiag(confusion_matrix)

    return confusion_matrix, confusion_matrix_diag


def getConfusionMatrix(predictions: pd.Series, actual: pd.Series):
    predictions, actual = evalPreprocessing(predictions, actual)

    return pd.crosstab(index=predictions,
                       columns=actual,
                       rownames=['Predicted'],
                       colnames=['Actual'],
                       dropna=False)


def getConfusionMatrixDiag(confusion_matrix: pd.DataFrame, actual_classes_in_index: bool = False):
    confusion_matrix_diag = {}

    if actual_classes_in_index:
        actual_classes = confusion_matrix.index
    else:
        actual_classes = confusion_matrix.columns

    for elem in actual_classes:
        try:
            confusion_matrix_diag[elem] = confusion_matrix.loc[elem, elem]
        except KeyError:
            continue

    return pd.Series(confusion_matrix_diag)


def computeAccuracy(predictions: pd.Series,
                    actual: pd.Series):
    confusion_matrix, confusion_matrix_diag = getConfMatrix_Diag(predictions, actual)
    nr_examples = confusion_matrix.sum().sum()
    nr_correctly_classified = confusion_matrix_diag.sum()

    return nr_correctly_classified / nr_examples


# TODO Fix TN and assure correctness in multi-class applications
def get_evaluation_metrics(predictions: pd.Series, actual: pd.Series, averaged: bool = True):
    """Returns four common confusion matrix metrics.

    If the parameter averaged is True, the method calculates the corresponding rates by averaging the vectors
    (note that this only applies to multi-class scenarios).
    Else, the vectors are returned as pandas.Series.

    :param predictions:     Predicted classes for data set
    :param actual:          Actual classes for data set
    :param averaged: Specifies whether metrics should be averaged or returned as vectors (only relevant for multi-class scenarios).
    :returns: FP (false positive), FN (false negative), TP (true positive), TN (true negative).
    """
    confusion_matrix, confusion_matrix_diag = getConfMatrix_Diag(predictions, actual)

    FP = confusion_matrix.sum(axis=1) - confusion_matrix_diag
    FN = confusion_matrix.sum(axis=0) - confusion_matrix_diag
    TP = confusion_matrix_diag
    # TN for each class is the sum of all values of the confusion matrix excluding that class's row and column (?)
    # or TN = confusion_matrix.sum() - (FP + FN + TP)
    TN = confusion_matrix.sum().sum() - confusion_matrix.sum(axis=0) - confusion_matrix.sum(axis=1)

    if averaged:
        FP = FP.sum() / len(confusion_matrix_diag)
        FN = FN.sum() / len(confusion_matrix_diag)
        TP = TP.sum() / len(confusion_matrix_diag)
        TN = TN.sum() / len(confusion_matrix_diag)

    return FP, FN, TP, TN


def computeRecall(predictions: pd.Series, actual: pd.Series):
    return computeTruePositiveRate(predictions, actual)


def computeTruePositiveRate(predictions: pd.Series, actual: pd.Series, averaged: bool = True):
    """Returns TPR = sum(True positives) / sum(Condition positive)."""
    confusion_matrix, confusion_matrix_diag = getConfMatrix_Diag(predictions, actual)

    tpr = confusion_matrix_diag / confusion_matrix.sum(axis=0)

    if averaged:
        tpr = tpr.sum() / len(confusion_matrix_diag)

    return tpr


# TODO Fix computation of False Positive Rate
def computeFalsePositiveRate(predictions: pd.Series, actual: pd.Series, averaged: bool = True):
    """Returns FPR = sum(False positives) / sum(Condition negative)."""
    confusion_matrix, confusion_matrix_diag = getConfMatrix_Diag(predictions, actual)

    false_positive_rate = 0
    for i in range(0, len(confusion_matrix_diag)):
        column_sum = confusion_matrix.iloc[:, i].sum()  # amount of examples actually in category
        classified_correctly = confusion_matrix_diag[i]
        classified_incorrectly = column_sum - classified_correctly

        false_positive_rate += classified_incorrectly / column_sum
    false_positive_rate /= len(confusion_matrix_diag)  # for average

    print("FPR (calc) = ", false_positive_rate)

    FP, FN, TP, TN = get_evaluation_metrics(predictions, actual, False)

    condition_negative = FP + TN
    return FP / condition_negative


def computePrecision(predictions: pd.Series, actual: pd.Series, averaged: bool = True):
    confusion_matrix, confusion_matrix_diag = getConfMatrix_Diag(predictions, actual)

    precision = confusion_matrix_diag / confusion_matrix.sum(axis=1)

    if averaged:
        precision = precision.sum() / len(confusion_matrix_diag)

    return precision


def f1_score(precision, recall):
    f1 = (2 * (precision * recall)) / (precision + recall)
    return f1
