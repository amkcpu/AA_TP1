import pandas
import numpy as np

from ml_tps.utils.probability_utils import relative_frequency as fr
from ml_tps.utils.dataframe_utils import subdateframe

class DecisionTree:

    def __init__(self, dataset: pandas.DataFrame, objective:str, nodes: int = None, variables: int = None):
        self.root = generate_subtree(dataset, objective)
        assa=4

    def classify(self, case):
        pass

    def plot(self):
        pass


def generate_subtree(dataset: pandas.DataFrame, objective: str):
    classes = list(dataset.keys())
    if len(classes) == 1:
        return dataset[objective].mode()[0]
    classes.remove(objective)
    if len(dataset[objective].unique()) == 1:
        return dataset[objective].unique()
    gain_list = np.array([gain(dataset=dataset,attribute=attr,objective=objective) for attr in classes])
    winner = classes[np.where(gain_list == np.amax(gain_list))[0][0]]
    values = dataset[winner].unique()
    subnodes = [generate_subtree(dataset=subdateframe(dataset,winner,v), objective=objective) for v in values]
    return (winner,dict(zip(values,subnodes)))


def shannon_entropy(dataset: pandas.DataFrame, objective: str):
    # H(S) = -p(+) * log2(p(+)) - p(-) * log2(p(-))
    # if p+ = 0  then (-p(+) * log2(p(+))) is 0
    ## General
    # f(x = p(+)) = - x * log2(x) if x != 0 else 0
    # H(S) = sum( f(x) for x in values)
    import math
    f = lambda x: -x * math.log2(x) if x != 0 else 0
    frs = fr(dataset[objective])
    #As dataset is an argument Sv is a subset of S
    return sum([f(x) for x in frs.values()])


def sv(dataset: pandas.DataFrame, attribute: str, value) -> pandas.DataFrame:
    return dataset[dataset[attribute] == value]


def gain(dataset: pandas.DataFrame, attribute: str, objective: str):
    svs = [sv(dataset, attribute, v) for v in dataset[attribute].unique()]
    general_entropy = shannon_entropy(dataset, objective=objective)
    return general_entropy - sum(len(_sv) / len(dataset) * shannon_entropy(_sv,objective) for _sv in svs)


def pour_titanic_dataset(dataset: pandas.DataFrame):
    dataset = dataset[["Pclass", "Survived", "Sex", "Age"]]

    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].dropna().mean())

    return dataset


if __name__ == '__main__':
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    DATA_FILEPATH_DEFAULT = f"{dir_path}/../tp2/data/tennis.tsv"
    dataset = pandas.read_csv(DATA_FILEPATH_DEFAULT,sep="\t")
    dataset = dataset.drop("Day", axis=1)
    d = DecisionTree(dataset,"Juega")

    DATA_FILEPATH_DEFAULT2 = f"{dir_path}/../tp2/data/titanic.csv"
    dataset2 = pandas.read_csv(DATA_FILEPATH_DEFAULT2,sep="\t")
    dataset2 = pour_titanic_dataset(dataset2)
    d2 = DecisionTree(dataset2,"Survived")
    asd = 5