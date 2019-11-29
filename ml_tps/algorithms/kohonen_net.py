import numpy as np
import pandas as pd
from typing import List
from matplotlib import pyplot as plt


def row_to_array(row: pd.Series):
    return np.array(row.values)


class KohonenPerceptron:

    def __init__(self, df: pd.DataFrame):
        self.w = row_to_array(df.iloc[np.random.randint(0, len(df.index) - 1)])

    def get_value(self, pattern: pd.Series):
        return np.dot(self.w, row_to_array(pattern))

    def update_values(self, coef, pattern: pd.Series):
        self.w += coef * (row_to_array(pattern) - self.w)


class KohonenNet:

    def __init__(self):
        self.perceptrons: List[KohonenPerceptron] = []
        self.side = 0
        self.generations = 0
        self.alpha = 0.5

    def pick_winner(self, row: pd.Series):
        i = 0
        j = 0
        max_value = self.perceptrons[0].get_value(row)

        for i_ in range(self.side):
            for j_ in range(self.side):
                value = self.perceptrons[i_ * self.side + j_].get_value(row)
                if max_value < value:
                    i = i_
                    j = j_
                    max_value = value
        return i, j

    def sigma(self):
        return 1/self.generations

    def eta(self):
        return self.generations ** -self.alpha

    def update(self, win_i: int, win_j: int, row: pd.Series):
        for i in range(self.side):
            for j in range(self.side):
                lambda_ = np.e ** -(((win_i - i)**2 + (win_j - j)**2) / (2 * self.sigma()**2))
                self.perceptrons[i * self.side + j].update_values(self.eta() * lambda_, row)

    def fit(self, df: pd.DataFrame, side: int, min_eta = 0.1, alpha = 0.5):
        """Fits the Kohonen model and sets the class variables.

        :param X: Data set to be clustered.
        :param side: Specifies the side of the square Kohonen net.
        """
        self.perceptrons = [KohonenPerceptron(df) for _ in range(side*side)]
        self.side = side
        self.generations = 1
        self.alpha = alpha

        stop = False
        while not stop:
            #          shuffle
            np.random.shuffle(df.values)
            for idx, row in df.iterrows():
                win_i, win_j = self.pick_winner(row)
                self.update(win_i,win_j,row)
                self.generations += 1
            stop = self.eta() < min_eta # OR no changes
            print("Current eta = ", self.eta())

    def predict(self, data: pd.DataFrame):
        """Assigns each given example to the nearest net in the previously fitted model.

        :param data: Data to be assigned using the previously fitted model.
        :return: Series containing the index number of each example's nearest cluster.
        """
        if self.perceptrons is []:
            raise ValueError("Model has not been fitted yet.")

        predictions = np.zeros((self.side, self.side))
        for idx, row in data.iterrows():
            i, j = self.pick_winner(row)
            predictions[i][j] += 1

        return predictions

    def plot(self, data: pd.DataFrame, objective: str) -> None:
        """Plots the Kohonen map for all available classes as well as the summed activations for all classes.

        :param data: Data to be predicted. Has to include an objective column containing the class label for each row.
        :param objective: Name of the column to be interpreted as class label.
        """
        classes = data[objective].unique()

        no_classes = len(classes) + 1           # +1 to account for image with summed activations.
        no_columns = 4
        no_rows = int(np.ceil(no_classes/no_columns))

        fig, ax = plt.subplots(no_rows, no_columns, sharey=True)
        fig.tight_layout()

        summed_activations = pd.DataFrame(np.zeros([self.side, self.side]))

        i = 0
        j = 0
        for cl in classes:
            predictions = self.predict(data[data[objective] == cl].drop(objective, axis=1))
            ax[i, j].set_title(cl)
            ax[i, j].imshow(predictions, cmap="Greys")
            summed_activations += predictions

            j += 1
            if j >= no_columns:
                j = 0
                i += 1

        ax[i, j].set_title("All classes", fontweight="bold")
        ax[i, j].imshow(summed_activations, cmap="Greys")
        plt.show()
