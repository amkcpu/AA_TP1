import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


class DistanceMetric:

    def __init__(self, metric: str, pass_dataframe: bool = False):
        """Distance between two points using different distance metrics.

        :param metric: Distance metric to be used. Supports Euclidean ("euclidean", "l2") and Manhattan ("manhattan", "l1)."""
        metrics = {"Euclidean": ["euclidean", "l2"],
                   "Manhattan": ["manhattan", "l1"]}

        if metric in metrics["Euclidean"]:
            self.metric = "euclidean"
        elif metric in metrics["Manhattan"]:
            self.metric = "manhattan"
        else:
            raise ValueError('"{0}" is not a supported metric for calculating cluster distances. '
                                 'The following dictionary lists the supported metrics as keys, '
                                 'and the corresponding keywords as values: {1}.'.format(metric, metrics))

    def calculate(self, x1: pd.Series, x2: pd.Series) -> float:
        if not (is_numeric_dtype(x1) and is_numeric_dtype(x2)):
            raise ValueError("Passed vectors not of numeric data type.")
        if len(x1) != len(x2):
            raise ArithmeticError("Vectors must have same length.")

        if self.metric == "euclidean":
            return euclidean_distance(x1, x2)
        else:
            return manhattan_distance(x1, x2)

    def calculate_df(self, df: pd.DataFrame, point: pd.Series) -> pd.Series:
        if not is_numeric_dtype(point) and ([is_numeric_dtype(df[col]) for col in df.columns].count(False) != 0):
            raise ValueError("At least one attribute is not of numeric data type.")
        if len(df.columns) != len(point):
            raise ArithmeticError("Number of attributes mismatch.")

        if self.metric == "euclidean":
            return euclidean_distance_dataframe(df, point)
        else:
            return manhattan_distance_dataframe(df, point)


def euclidean_distance(x1: pd.Series, x2: pd.Series) -> float:
    """Metric based on squared distances. Also known as L2 or 2-Norm."""
    return np.sqrt(sum(np.square(x1 - x2)))


def euclidean_distance_dataframe(df: pd.DataFrame, point: pd.Series) -> pd.Series:
    """Metric based on squared distances. Also known as L2 or 2-Norm."""
    return np.sqrt(np.square(df - point).sum(axis=1))


def manhattan_distance(x1: pd.Series, x2: pd.Series) -> float:
    """Metric that sums the absolute distances. Also known as taxicab geometry metric, L1 or 1-Norm."""
    return sum(np.abs(x1 - x2))


def manhattan_distance_dataframe(df: pd.DataFrame, point: pd.Series) -> pd.Series:
    """Metric that sums the absolute distances. Also known as taxicab geometry metric, L1 or 1-Norm."""
    return np.abs(df - point).sum(axis=1)
