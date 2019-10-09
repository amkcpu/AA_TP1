import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def fit_using_normal_equation(X: pd.DataFrame, y: pd.Series, plot: bool = False):
    X_copy = X.copy()
    add_bias_to_dataset(X_copy)

    b = np.linalg.inv(X_copy.T @ X_copy) @ (X_copy.T @ y)       # b = inv(X'X)*X'y

    if plot:
        plot_2d_data_regression_line(X, y, b)

    return b


def add_bias_to_dataset(dataset: pd.DataFrame):
    ones = pd.Series(np.ones(max(dataset.index) + 1))
    dataset.insert(0, "Bias", ones)         # works inplace


def plot_2d_data_regression_line(X: pd.DataFrame, y: pd.Series, b):
    x_sampling = pd.DataFrame(np.linspace(min(X.values), max(X.values), 1000))

    plt.plot(X.values, y.values, "bo")
    plt.plot(x_sampling, predict(x_sampling, b), "r")
    plt.show()


def predict(X: pd.DataFrame, b):
    X_copy = X.copy()
    add_bias_to_dataset(X_copy)

    return X_copy @ b
