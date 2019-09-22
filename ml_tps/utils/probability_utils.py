import pandas as pd
import numpy as np

def control_probability(value):
    if value < 0 or 1 < value:
        raise ValueError("Probability out of range [0,1]")

def absolute_frequency(array: np.ndarray):
    u, c = np.unique(array,return_counts=True)
    return dict(zip(u,c))

def relative_frequency(array: np.ndarray):
    rel_frequency = dict()
    for k, v in absolute_frequency(array).items():
        rel_frequency[k] = v / array.size
    return rel_frequency

def confussion_matrix(classifier, test_dataset: pd.DataFrame, objective: str):
    mat = pd.DataFrame({"0": [0, 0], "1": [0, 0], "-1": [0, 0]})
    for case in test_dataset.iterrows():
        case = case[1]
        ans = classifier.classify(case)
        mat.loc[case[objective]][ans] += 1
    return mat