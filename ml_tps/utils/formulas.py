import numpy as np
import pandas as pd
import math
from ml_tps.utils.probability_utils import relative_frequency


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def shannon_entropy(dataset: pd.DataFrame, objective: str):
    # H(S) = -p(+) * log2(p(+)) - p(-) * log2(p(-))
    # if p+ = 0  then (-p(+) * log2(p(+))) is 0
    ## General
    # f(x = p(+)) = - x * log2(x) if x != 0 else 0
    # H(S) = sum( f(x) for x in values)
    f = lambda x: -x * math.log2(x) if x != 0 else 0
    frs = relative_frequency(dataset[objective])
    # As data set is an argument Sv is a subset of S
    return sum([f(x) for x in frs.values()])


def gini_index(dataset: pd.DataFrame, objective: str):
    return 1 - sum(relative_frequency(dataset[objective]))
