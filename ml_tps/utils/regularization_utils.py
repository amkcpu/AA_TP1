import pandas as pd
import numpy as np


class Regularization:

    def __init__(self, reg_type: str = None, lam: float = 0.01):
        """Provides regularization for a multitude of algorithms.

        :param reg_type: Type of regularization to be used. Can use Ridge/L2 ("l1), Lasso/L1 ("l2") or return zero (equals no regularization).
        :param lam: Regularization strength parameter commonly known as Lambda.
        """
        reg_types = {"Lasso": ["lasso", "l1", "absolute"],
                     "Ridge": ["ridge", "l2", "squared"],
                     "None/zero": ["none", "zero"]}

        if reg_type in reg_types["Lasso"]:
            self.method = l1_regularization
        elif reg_type in reg_types["Ridge"]:
            self.method = l2_regularization
        elif (reg_type is None) or (reg_type in reg_types["None/zero"]):
            self.method = zero_regularization
        else:
            raise ValueError(
                '"{0}" is not a supported regularization method. The following dictionary lists the supported '
                'methods as keys, and the corresponding keywords as values: {1}.'.format(reg_type, reg_types))

        self.lam = lam

    def calculate(self, theta: pd.DataFrame) -> float:
        """Calculates the regularization.

        :param theta: Model parameters.
        :return: Regularization term for parameter estimation.
        """
        return self.method(theta, self.lam)


def l1_regularization(theta: pd.DataFrame, lam: float) -> float:
    """L1 regularization which uses absolute values. ALso known as L1 or Lasso."""
    return lam * np.abs(theta).sum().sum()


def l2_regularization(theta: pd.DataFrame, lam: float) -> float:
    """L2 regularization which uses squared values. Also known as L2 or Ridge."""
    return lam * np.square(theta).sum().sum()


def zero_regularization(theta: pd.DataFrame, lam: float) -> float:
    return 0
