import numpy as np


def control_probability(value):
    if value < 0 or 1 < value:
        raise ValueError("Probability out of range [0,1]")


def absolute_frequency(array: np.ndarray):
    u, c = np.unique(array, return_counts=True)
    return dict(zip(u, c))


def relative_frequency(array: np.ndarray):
    rel_frequency = dict()
    for k, v in absolute_frequency(array).items():
        rel_frequency[k] = v / array.size
    return rel_frequency
