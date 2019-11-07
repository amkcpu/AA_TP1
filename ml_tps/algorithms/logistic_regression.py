import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_tps.utils.formulas import sigmoid
from ml_tps.utils.data_processing import add_bias_to_dataset
from ml_tps.utils.regularization_utils import Regularization


class LogisticRegression:

    def __init__(self, initial_theta: pd.Series = None):
        """Implements Logistic Regression assuming two prediction classes, 0 and 1.

        :param initial_theta: Initial values for theta. Has to include bias term."""
        self.theta = initial_theta
        if initial_theta is not None:
            self.theta.index = range(0, len(initial_theta))

    def fit(self, X: pd.DataFrame, y: pd.Series, alpha: float = 0.5, max_iter: int = 500, tol: float = 0.001,
            reg_type: str = None, lam: float = 1.0, plot_cost_vs_iterations: bool = False):
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

        reg = Regularization(reg_type, lam)
        n = len(X)
        X_biased = add_bias_to_dataset(X, reset_columns=True)
        it = 0
        errors = []
        error = self.cost(X, y)
        while it < max_iter and error > tol:
            errors.append(error)
            reg_term = reg.calculate(self.theta)

            h = sigmoid(X_biased @ self.theta)
            grad = (X_biased.T @ (h - y)) + reg_term
            self.theta -= (alpha/n) * grad

            error = self.cost(X, y)
            it += 1

        print("Finished after {} iterations.".format(it))
        print("Converged with error (cost) = {}.".format(error))

        if plot_cost_vs_iterations:
            plt.plot(range(0, it), errors, linewidth=1)
            plt.xlabel("No. of iterations")
            plt.ylabel("Cost")
            plt.show()

    def predict(self, X: pd.DataFrame, return_as_probabilities: bool = False):
        """ Predict using trained logistic regression parameters.

        :param X: pandas.DataFrame of values to be predicted.
        :return predictions: pandas.Series with predictions for each example in given DataFrame.
        """
        if self.theta is None:
            raise ValueError("Model has not been fitted yet (regression parameters theta = None).")

        X_biased = add_bias_to_dataset(X, reset_columns=True)
        predictions = sigmoid(X_biased @ self.theta)
        if return_as_probabilities:
            return predictions
        else:
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
        h = sigmoid(X_biased @ self.theta)
        return -y.T @ np.log(h) - ((1 - y.T) @ np.log(1 - h)) / n
