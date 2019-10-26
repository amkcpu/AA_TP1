import pandas as pd
import numpy as np
from ml_tps.utils.formula_utils import sigmoid
from ml_tps.utils.dataframe_utils import add_bias_to_dataset


class LogisticRegression:
    """Implements Logistic Regression assuming two prediction classes, 0 and 1."""

    def __init__(self, initial_theta: pd.Series = None):
        """:param initial_theta: Initial values for theta. Has to include bias term."""
        self.theta = initial_theta

    def fit(self, X: pd.DataFrame, y: pd.Series, alpha: float=0.0001, iters: int=1000, tol: float = 0.001):
        """Fit logistic regression parameters using gradient descent.
        
        :param X: Training set examples.
        :param y: Training set objective values.
        :param alpha: Specifies learning rate. Defaults to 0.0001.
        :param iters: Number of iterations of gradient descent algorithm. Defaults to 1000 iterations.
        :param tol: Maximum error tolerance.
        """
        if self.theta is None:
            self.theta = pd.Series(np.zeros(len(X.columns) + 1))

        it = 0
        error = self.cost(X, y)
        while it < iters and error > tol:
            theta_update = alpha * (X.T @ (sigmoid(X @ self.theta) - y))
            bias_update = alpha * sum(sigmoid(X @ self.theta) - y)

            self.theta -= bias_update.append(theta_update)

            error = self.cost(X, y)
            it += 1

    def predict(self, X: pd.DataFrame):
        """ Predict using trained logistic regression parameters.

        :param X: pandas.DataFrame of values to be predicted.
        :return predictions: pandas.Series with predictions for each example in given DataFrame.
        """
        if self.theta is None:
            raise ValueError("Model has not been fitted yet (regression parameters theta = None).")

        index = X.index
        X = add_bias_to_dataset(X)
        predictions = sigmoid(X @ self.theta)
        predictions = predictions.apply(lambda x: 0 if x < 0.5 else 1)
        predictions.index = index
        return predictions

    def cost(self, X, y):
        """Logistic regression cost function.

        :param X: Given point.
        :param y: Given class for example point.
        :return: Cost of one single example using trained theta parameters.
        """
        if self.theta is None:
            raise ValueError("Model has not been fitted yet (regression parameters theta = None).")

        X = add_bias_to_dataset(X)
        return -y.T * np.log(sigmoid(X @ self.theta)) - (1 - y).T * np.log(1 - sigmoid(X @ self.theta))