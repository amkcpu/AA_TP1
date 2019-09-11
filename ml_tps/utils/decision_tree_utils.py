import pandas
from ml_tps.utils.probability_utils import relative_frequency as fr

class DecisionTree:

    def __init__(self, dataset: pandas.DataFrame, nodes: int = None, variables: int = None):
        pass

    def classify(self, case):
        pass

    def plot(self):
        pass


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


if __name__ == '__main__':
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    DATA_FILEPATH_DEFAULT = f"{dir_path}/../tp2/data/tennis.tsv"
    dataset = pandas.read_csv(DATA_FILEPATH_DEFAULT,sep="\t")
    shannon_entropy(dataset=dataset, objective="Juega")
    aa = sv(dataset,"Viento","weak")
    shannon_entropy(dataset=aa, objective="Juega")
    ggg = gain(dataset,"Viento", "Juega")
    asd = 5