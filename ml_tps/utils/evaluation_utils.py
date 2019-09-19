import pandas as pd


# F1-score
def f1_score(precision, recall):
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# TODO this actually contains an error occurring in small validation samples,
#  whereby some categories might not be predicted and the confusion matrix might
#  not by square. An error gets thrown when initializing confusion_matrix_diag, but
#  already the confusion_matrix itself is faulty.
#  Using a large sample, this problem is avoided.
def confusion_matrix(predictions: pd.Series, actual: pd.Series):
    return pd.crosstab(predictions,
                       actual,
                       rownames=['Actual'],
                       colnames=['Predicted'])
