import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def fit_using_normal_equation(X: pd.DataFrame, y: pd.Series):
    X = add_bias_to_dataset(X)
    return np.linalg.inv(X.T @ X) @ (X.T @ y)       # b = inv(X'X)*X'y


def add_bias_to_dataset(dataset: pd.DataFrame):
    ones = pd.Series(np.ones(max(dataset.index) + 1))
    dataset_copy = dataset.copy()
    dataset_copy.insert(0, "Bias", ones)         # works inplace

    return dataset_copy


def plot_2d(X: pd.DataFrame, y: pd.Series, b):
    if len(X.columns) != 1:
        raise ValueError("This method only plots 2D data and regression lines. "
                         "The X that was passed has {} columns instead of 1.".format(len(X.columns)))

    x_sampling = pd.DataFrame(np.linspace(min(X.values), max(X.values), 1000))

    plt.plot(X.values, y.values, "bo")
    plt.plot(x_sampling, predict(x_sampling, b), "r")
    plt.show()


def predict(X: pd.DataFrame, b):
    X = add_bias_to_dataset(X)

    return X @ b
