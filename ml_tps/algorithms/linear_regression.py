import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ml_tps.utils.data_processing import add_bias_to_dataset


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

    def plot(self, X: pd.DataFrame, y: pd.Series):
        no_predictors = len(X.columns)

        if no_predictors == 1:
            sampling_data = pd.DataFrame(np.linspace(min(X.values), max(X.values), 1000))

            plt.plot(X, y, "bo")
            plt.plot(sampling_data, self.predict(sampling_data), "r")
            plt.show()
        elif no_predictors == 2:        # TODO fix 3D plot
            sampling_data = pd.DataFrame(np.linspace(min(X.iloc[:, 0].values), max(X.iloc[:, 0].values), 1000))
            sampling_data[1] = pd.DataFrame(np.linspace(min(X.iloc[:, 1].values), max(X.iloc[:, 1].values), 1000))

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter(X.iloc[:, 0].values, X.iloc[:, 1].values, y.values, color="blue")
            ax.plot_trisurf(sampling_data[0], sampling_data[1], self.predict(sampling_data), color="red")
            plt.show()
        else:
            raise ValueError("This method only plots 2D and 3D data and regression hyperplanes. "
                             + "The X that was passed has {} columns instead of 1 or 2.".format(no_predictors))

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
