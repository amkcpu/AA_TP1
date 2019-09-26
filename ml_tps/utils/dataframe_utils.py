import numpy as np
import pandas as pd
import numbers
from sklearn.preprocessing import MinMaxScaler


def subdataframe(dataset: pd.DataFrame, attribute: str, value = None):
    if value is None:
        return [subdataframe(dataset, attribute=attribute, value=value) for value in dataset[attribute].unique()]
    return dataset[dataset[attribute] == value].drop(attribute, axis=1)


def subdataframe_with_repeated(dataset: pd.DataFrame, size: int):
    return pd.DataFrame([dataset.iloc[i] for i in np.random.randint(len(dataset), size=size)]).reset_index(drop=True)


def divide_in_training_test_datasets(dataset: pd.DataFrame, train_pctg: float = 0.5):
    indexes = np.arange(len(dataset))
    np.random.shuffle(indexes)
    indexes = list(indexes)
    split_index = int(np.floor(len(dataset) * train_pctg))
    train = pd.DataFrame([dataset.iloc[i] for i in indexes[:split_index]])
    test = pd.DataFrame([dataset.iloc[i] for i in indexes[split_index:]])
    return train, test

# Rewrite negatives and positives to 0 and 1
def rewritePositivesNegatives(dataset: pd.DataFrame):
    for i in dataset.columns:
        dataset[i] = dataset[i].apply(lambda x: 1 if x == "positive" else 0 if x == "negative" else x)
    return dataset


# Based on check whether first element in column is Number or not
def deleteNonNumericColumns(dataset: pd.DataFrame):
    for i in dataset.columns:
        first_element_in_column = dataset[i].iloc[0]
        if not isinstance(first_element_in_column, numbers.Number):
            del dataset[i]
    return dataset


# Scales given dataset, ignoring objective column
def scale_dataset(dataset: pd.DataFrame, objective: str, ignore_objective: bool=True, scaling_type: str="minmax"):
    if scaling_type == "minmax":    # TODO add more/different scaling types
        scaler = MinMaxScaler()
    else:
        scaler = MinMaxScaler()

    if ignore_objective:
        X, y = seperateDatasetObjectiveData(dataset, objective)
        X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
        return pd.concat([X_scaled, y], axis=1)
    else:
        return pd.DataFrame(scaler.fit_transform(dataset), index=dataset.index, columns=dataset.columns)


def seperateDatasetObjectiveData(dataset: pd.DataFrame, objective: str):
    X = dataset.loc[:, dataset.columns != objective]
    y = dataset[objective]

    return X, y
