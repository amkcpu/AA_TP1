import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def euclidean_distance(x1: pd.Series, x2: pd.Series):
    if not (is_numeric_dtype(x1) and is_numeric_dtype(x2)):
        raise ValueError("Euclidean Distance: Passed vectors not of numeric data type.")
    if len(x1) != len(x2):
        raise ArithmeticError("Euclidean Distance: Vectors must have same length.")

    return np.sqrt(sum(np.square(x1 - x2)))


def manhattan_distance(x1: pd.Series, x2: pd.Series):
    """Metric that sums the absolute distances. Also known as taxicab geometry metric or 1-Norm."""
    if not (is_numeric_dtype(x1) and is_numeric_dtype(x2)):
        raise ValueError("Manhattan Distance: Passed vectors not of numeric data type.")
    if len(x1) != len(x2):
        raise ArithmeticError("Manhattan Distance: Vectors must have same length.")

    return sum(np.abs(x1 - x2))
