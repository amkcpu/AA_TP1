import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def add_bias_to_dataset(dataset: pd.DataFrame):
    ones = pd.Series(np.ones(max(dataset.index) + 1))
    dataset_copy = dataset.copy()
    dataset_copy.insert(0, "Bias", ones)  # works inplace

    return dataset_copy


class LinearRegression:

    def __init__(self, initial_b: pd.Series = None):
        self.b = initial_b

    def fit(self, X: pd.DataFrame, y: pd.Series):       # using Normal Equation approach
        X = add_bias_to_dataset(X)
        self.b = np.linalg.inv(X.T @ X) @ (X.T @ y)     # b = inv(X'X)*X'y

    def predict(self, X: pd.DataFrame):
        if self.b is None:
            raise ValueError("Model has not been fitted yet (regression parameters b = None).")
        X = add_bias_to_dataset(X)

        return X @ self.b

    def plot_2d(self, X: pd.DataFrame, y: pd.Series):
        if len(X.columns) != 1:
            raise ValueError("This method only plots 2D data and regression lines. "
                             "The X that was passed has {} columns instead of 1.".format(len(X.columns)))

        x_sampling = pd.DataFrame(np.linspace(min(X.values), max(X.values), 1000))

        plt.plot(X.values, y.values, "bo")
        plt.plot(x_sampling, self.predict(x_sampling), "r")
        plt.show()

    def calculate_sums_of_squares(self, X: pd.DataFrame, y: pd.Series):
        TSS = (y - y.mean()).T @ (y - y.mean())     # TSS = (y-mean(y)'(y-mean(y)
        errors = y - self.predict(X)                # e = y - Xb
        RSS = errors.T @ errors                     # RSS = e'e
        ESS = TSS - RSS                             # ESS = TSS - RSS

        return TSS, ESS, RSS

    def calculate_r2(self, X: pd.DataFrame, y: pd.Series):
        TSS, ESS, RSS = self.calculate_sums_of_squares(X, y)

        return ESS / TSS

    def calculate_adjusted_r2(self, X: pd.DataFrame, y: pd.Series):
        R2 = self.calculate_r2(X, y)
        no_examples = len(X.index)
        no_predictors = len(X.columns)

        return 1 - (1 - R2) * ((no_examples - 1) / (no_examples - no_predictors - 1))
