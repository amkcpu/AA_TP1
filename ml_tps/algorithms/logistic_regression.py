import pandas as pd
import numpy as np
from ml_tps.utils.formulas import sigmoid, l1_regularization, l2_regularization
from ml_tps.utils.data_processing import add_bias_to_dataset


class LogisticRegression:
    """Implements Logistic Regression assuming two prediction classes, 0 and 1."""

    def __init__(self, initial_theta: pd.Series = None):
        """:param initial_theta: Initial values for theta. Has to include bias term."""
        self.theta = initial_theta
        if initial_theta is not None:
            self.theta.index = range(0, len(initial_theta))

    def fit(self, X: pd.DataFrame, y: pd.Series, alpha: float = 0.0001, max_iter: int = 1000, tol: float = 0.001,
            reg_type: str = None, lam: float = 1.0):
        """Fit logistic regression parameters using gradient descent.
        
        :param X: Training set examples.
        :param y: Training set objective values.
        :param alpha: Specifies learning rate. Defaults to 0.0001.
        :param max_iter: Number of iterations of gradient descent algorithm. Defaults to 1000 iterations.
        :param tol: Maximum error tolerance.
        :param reg_type: Specify type of regularization to be applied. Supports L1 (absolute) "l1"
                        and L2 (squared) "l2" regularization.
        :param lam: Regularization strength parameter commonly known as Lambda.
        """
        if self.theta is None:
            self.theta = pd.Series(np.zeros(len(X.columns) + 1))

        if reg_type == "l1":
            reg_method = l1_regularization
        elif reg_type == "l2":
            reg_method = l2_regularization
        elif reg_type is None:
            reg_method = lambda theta, lam: 0  # always map to 0

        X_biased = add_bias_to_dataset(X, reset_columns=True)
        it = 0
        error = self.cost(X, y)
        while it < max_iter and error > tol:
            reg_term = reg_method(self.theta, lam)

            theta_update = alpha * (X_biased.T @ (sigmoid(X_biased @ self.theta) - y)) + reg_term
            bias_update = alpha * sum(sigmoid(X_biased @ self.theta) - y)

            self.theta -= pd.Series(bias_update).append(theta_update, ignore_index=True)

            error = self.cost(X, y)
            it += 1

        print("Finished after {} iterations.".format(it))
        print("Converged with error (cost) = {}.".format(error))

    def predict(self, X: pd.DataFrame):
        """ Predict using trained logistic regression parameters.

        :param X: pandas.DataFrame of values to be predicted.
        :return predictions: pandas.Series with predictions for each example in given DataFrame.
        """
        if self.theta is None:
            raise ValueError("Model has not been fitted yet (regression parameters theta = None).")

        X_biased = add_bias_to_dataset(X, reset_columns=True)
        predictions = sigmoid(X_biased @ self.theta)
        return predictions.apply(lambda x: 0 if x < 0.5 else 1)

    def cost(self, X: pd.DataFrame, y: pd.Series):
        """Logistic regression cost function.

        :param X: Training set examples.
        :param y: Training set objective values.
        :return: Cost of given examples using trained theta parameters.
        """
        if self.theta is None:
            raise ValueError("Model has not been fitted yet (regression parameters theta = None).")

        n = len(X)
        X_biased = add_bias_to_dataset(dataset=X, reset_columns=True)
        return -y.T @ np.log(sigmoid(X_biased @ self.theta)) - (1 - y).T @ np.log(1 - sigmoid(X_biased @ self.theta)) / n
