# Distance metrics, regularization formulas, entropies
import numpy as np
import pandas as pd
import math
from pandas.api.types import is_numeric_dtype
from ml_tps.utils.probability_utils import relative_frequency


def euclidean_distance(x1: pd.Series, x2: pd.Series):
    if not (is_numeric_dtype(x1) and is_numeric_dtype(x2)):
        raise ValueError("Euclidean Distance: Passed vectors not of numeric data type.")
    if len(x1) != len(x2):
        raise ArithmeticError("Euclidean Distance: Vectors must have same length.")

    return np.sqrt(sum(np.square(x1 - x2)))


def l2_distance(x1: pd.Series, x2: pd.Series):
    return euclidean_distance(x1, x2)


def manhattan_distance(x1: pd.Series, x2: pd.Series):
    """Metric that sums the absolute distances. Also known as taxicab geometry metric or 1-Norm."""
    if not (is_numeric_dtype(x1) and is_numeric_dtype(x2)):
        raise ValueError("Manhattan Distance: Passed vectors not of numeric data type.")
    if len(x1) != len(x2):
        raise ArithmeticError("Manhattan Distance: Vectors must have same length.")

    return sum(np.abs(x1 - x2))


def l1_distance(x1: pd.Series, x2: pd.Series):
    return manhattan_distance(x1, x2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def l1_regularization(theta: pd.DataFrame, lam: float):
    """L1 regularization which uses absolute values.

    :param theta: Model parameters.
    :param lam: Regularization strength parameter commonly known as Lambda.
    :return: Regularization term for parameter estimation.
    """
    return lam * np.abs(theta).sum().sum()


def l2_regularization(theta: pd.DataFrame, lam: float):
    """L2 regularization which uses squared values.

    :param theta: Model parameters.
    :param lam: Regularization strength parameter commonly known as Lambda.
    :return: Regularization term for parameter estimation.
    """
    return lam * np.square(theta).sum().sum()


def shannon_entropy(dataset: pd.DataFrame, objective: str):
    # H(S) = -p(+) * log2(p(+)) - p(-) * log2(p(-))
    # if p+ = 0  then (-p(+) * log2(p(+))) is 0
    ## General
    # f(x = p(+)) = - x * log2(x) if x != 0 else 0
    # H(S) = sum( f(x) for x in values)
    f = lambda x: -x * math.log2(x) if x != 0 else 0
    frs = relative_frequency(dataset[objective])
    # As data set is an argument Sv is a subset of S
    return sum([f(x) for x in frs.values()])


def gini_index(dataset: pd.DataFrame, objective: str):
    return 1 - sum(relative_frequency(dataset[objective]))
