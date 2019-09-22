import numpy as np
import pandas as pd
import numbers


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