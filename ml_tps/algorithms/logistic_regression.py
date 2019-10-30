import pandas as pd

class LogisticRegression:

    def __init__(self, initial_b: pd.Series = None):
        self.b = initial_b

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO using gradient descent?

    def predict(self, X: pd.DataFrame):
        if self.b is None:
            raise ValueError("Model has not been fitted yet (regression parameters b = None).")
        # TODO
