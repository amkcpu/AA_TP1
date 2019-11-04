import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from ml_tps.utils.data_processing import add_bias_to_dataset
from ml_tps.utils.regularization_utils import Regularization


def gradient_descent(X: pd.DataFrame, y: pd.Series, theta: pd.Series = None, alpha: float = 0.01, max_iter: int = 300,
                     tol: float = 0.01, reg_type: str = None, lam: float = 0.0001,
                     plot_cost_vs_iterations: bool = False) -> pd.Series:
    """Provides linear regression fitting using gradient descent.

    Optionally uses L1 or L2 regularization to prevent overfitting. For explanation of each parameter refer to the
    LinearRegression.fit() method.

    :returns: Calculated theta as Series."""
    reg = Regularization(reg_type, lam)
    X_biased = add_bias_to_dataset(X, reset_columns=True)
    n = len(X)
    it = 0
    errors = []
    error = cost(theta, X, y)
    while it < max_iter and error > tol:
        errors.append(error)
        reg_term = reg.calculate(theta)

        theta_update = (alpha / n) * (X_biased.T @ ((X_biased @ theta) - y)) + reg_term
        bias_update = (alpha / n) * sum(X_biased @ theta - y)

        theta -= pd.Series(bias_update).append(theta_update[1:], ignore_index=True)

        error = cost(theta, X, y)
        it += 1

    print("Finished after {} iterations.".format(it))
    print("Converged with error (cost) = {}.".format(error))

    if plot_cost_vs_iterations:
        plt.plot(range(0, it), errors, linewidth=1)
        plt.xlabel("No. of iterations")
        plt.ylabel("Cost")
        plt.show()

    return theta


def cost(theta: pd.Series, X: pd.DataFrame, y: pd.Series) -> float:
    """Linear regression cost function."""
    X_biased = add_bias_to_dataset(X, reset_columns=True)
    n = len(X)
    return 1 / (2 * n) * ((X_biased @ theta - y).T @ (X_biased @ theta - y))


def normal_equation(X: pd.DataFrame, y: pd.Series):
    """Provides linear regression fitting using the normal equation approach."""
    X = add_bias_to_dataset(X, reset_columns=True)
    return pd.Series(np.linalg.inv(X.T @ X) @ (X.T @ y))  # b = inv(X'X)*X'y


class LinearRegression:

    def __init__(self, initial_b: pd.Series = None):
        self.b = initial_b

    def fit(self, X: pd.DataFrame, y: pd.Series, mode: str = "auto", alpha: float = 0.01, max_iter: int = 300,
            tol: float = 0.01, reg_type: str = None, lam: float = 0.0001,
            plot_cost_vs_iterations: bool = False) -> None:
        """Fits the parameters of a linear regression model.

        :param X: Sample values not including objective nor bias (is added automatically).
        :param y: Values for the model's objective.
        :param mode: Determines how the model is fit. Use "gradient" for gradient descent, "normal" for normal equation
                     and "auto" to choose gradient descent when the number of features is >1000, normal equation otherwise.
        :param alpha: Learning rate for gradient descent.
        :param max_iter: Maximum number of iterations for gradient descent.
        :param tol: Maximum error tolerance for gradient descent.
        :param reg_type: Type of regularization to be used. Can only be used in conjunction with gradient descent, otherwise it is ignored.
                        Can use Ridge/L2 ("l1) or Lasso/L1 ("l2").
        :param lam: Lambda parameter for regularization.
        :param plot_cost_vs_iterations: When using gradient descent, can plot the error in each iteration to check if gradient descent works as expected.
        """
        modes = {"Auto": ["auto"],
                 "Gradient Descent": ["gd", "gradient"],
                 "Normal Equation": ["norm", "normal"]}

        if mode in modes["Auto"]:
            if (len(X.columns) - 1) > 1000:
                self.b = gradient_descent(X=X, y=y, theta=self.b, alpha=alpha, max_iter=max_iter, tol=tol,
                                          reg_type=reg_type, lam=lam, plot_cost_vs_iterations=plot_cost_vs_iterations)
            else:
                self.b = normal_equation(X, y)
        elif mode in modes["Gradient Descent"]:
            self.b = gradient_descent(X=X, y=y, theta=self.b, alpha=alpha, max_iter=max_iter, tol=tol,
                                      reg_type=reg_type, lam=lam, plot_cost_vs_iterations=plot_cost_vs_iterations)
        elif mode in modes["Normal Equation"]:
            self.b = normal_equation(X, y)
        else:
            raise ValueError('"{0}" is not a supported method for fitting the linear regression model. '
                             'The following dictionary lists the supported methods as keys, '
                             'and the corresponding keywords as values: {1}.'.format(mode, modes))

    def predict(self, X: pd.DataFrame):
        no_predictors = len(self.b) - 1
        no_predictive_columns = len(X.columns)
        if self.b is None:
            raise ValueError("Model has not been fitted yet (regression parameters b = None).")
        if no_predictors != no_predictive_columns:
            raise ValueError("Passed data does not align with the fitted regression model."
                             "The model considers {0} predictors, whereas the data given assumes {1}."
                             .format(no_predictors, no_predictive_columns))
        X = add_bias_to_dataset(X, reset_columns=True)

        return X @ self.b

    def plot(self, X: pd.DataFrame, y: pd.Series, title: str = None) -> None:
        """Plot data as well as the hyperplane found by the model. Only for 2D or 3D problems.

        :param X: Data values to be plotted.
        :param y: Objective values.
        :param title: Can optionally specify a title for the plot.
        """
        no_predictors = len(self.b) - 1
        if no_predictors > 2:
            raise ValueError("This method only plots 2D and 3D data and regression hyperplanes. "
                             + "The X that was passed has {} columns instead of 1 or 2.".format(no_predictors))

        X.columns = range(0, len(X.columns))  # makes accessing column more easy if they have been renamed
        self.predict(X)  # in order to catch prediction errors

        if no_predictors == 1:
            sampling_data = pd.DataFrame(np.linspace(min(X[0]), max(X[0]), 1000))

            plt.plot(X, y, "bo")
            plt.plot(sampling_data, self.predict(sampling_data), "r")
            plt.title(title, fontweight="bold")
            plt.show()
        elif no_predictors == 2:
            x_space = np.linspace(min(X[0]), max(X[0]), 1000)
            y_space = np.linspace(min(X[1]), max(X[1]), 1000)
            X_mesh, Y_mesh = np.meshgrid(x_space, y_space)
            surface = self.b[0] + self.b[1] * X_mesh + self.b[2] * Y_mesh

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[0], X[1], y, color="blue")
            ax.plot_surface(X_mesh, Y_mesh, surface, color="red", alpha=0.5)
            plt.title(title, fontweight="bold")
            plt.show()

    def calculate_sums_of_squares(self, X: pd.DataFrame, y: pd.Series):
        TSS = (y - y.mean()).T @ (y - y.mean())  # TSS = (y-mean(y)'(y-mean(y)
        errors = y - self.predict(X)  # e = y - Xb
        RSS = errors.T @ errors  # RSS = e'e
        ESS = TSS - RSS  # ESS = TSS - RSS

        return TSS, ESS, RSS

    def calculate_r2(self, X: pd.DataFrame, y: pd.Series) -> float:
        TSS, ESS, RSS = self.calculate_sums_of_squares(X, y)

        return ESS / TSS

    def calculate_adjusted_r2(self, X: pd.DataFrame, y: pd.Series) -> float:
        R2 = self.calculate_r2(X, y)
        no_examples = len(X.index)
        no_predictors = len(X.columns)

        return 1 - (1 - R2) * ((no_examples - 1) / (no_examples - no_predictors - 1))
