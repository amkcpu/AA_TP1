import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from pandas.api.types import is_numeric_dtype


def subdataframe(dataset: pd.DataFrame, attribute: str, value=None):
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


def rewrite_positives_negatives(dataset: pd.DataFrame):
    """Rewrite "negative" and "positive" to 0 and 1, respectively."""
    for i in dataset.columns:
        dataset[i] = dataset[i].apply(lambda x: 0 if x == "negative" else 1 if x == "positive" else x)
    return dataset


def delete_non_numeric_columns(dataset: pd.DataFrame):
    for i in dataset.columns:
        if not is_numeric_dtype(dataset[i]):
            dataset = dataset.drop(i, axis="columns")
    return dataset


def scale_dataset(dataset: pd.DataFrame, scaling_type: str = None, objective: str = None):
    """Scales/normalizes a given dataset.

    :param dataset:         Data set to be scaled.
    :param scaling_type:    Scaling type used to normalize, defaulting to StandardScaler.
    :param objective:       If passed, the column with this name will not be scaled.
    :returns: Scaled data set as pandas.DataFrame.
    """
    if scaling_type == "minmax":
        scaler = MinMaxScaler()  # (X - X.min()) / (X.max() - X.min())
    elif scaling_type == "maxabs":
        scaler = MaxAbsScaler()  # X / abs(X.max())
    elif scaling_type == "robust":
        scaler = RobustScaler(quantile_range=(25.0, 75.0))  # X / X.quantile_range()
    else:
        scaler = StandardScaler(with_mean=True, with_std=True)  # (X - X.mean()) / X.std_deviation

    if objective is not None:
        X, y = separate_dataset_objective_data(dataset, objective)
        X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
        return pd.concat([X_scaled, y], axis=1)
    else:
        return pd.DataFrame(scaler.fit_transform(dataset), index=dataset.index, columns=dataset.columns)


def separate_dataset_objective_data(dataset: pd.DataFrame, objective: str):
    X = dataset.loc[:, dataset.columns != objective]
    y = dataset[objective]

    return X, y


def get_test_train_X_y(data: pd.DataFrame, objective: str, train_pctg: float = 0.5):
    train, test = divide_in_training_test_datasets(data, train_pctg)
    X_train, y_train = separate_dataset_objective_data(train, objective)
    X_test, y_test = separate_dataset_objective_data(test, objective)

    return X_train, y_train, X_test, y_test


def add_bias_to_dataset(dataset: pd.DataFrame):
    ones = pd.Series(np.ones(max(dataset.index) + 1))
    dataset_copy = dataset.copy()
    dataset_copy.insert(0, "Bias", ones)  # works inplace

    return dataset_copy


def drop_objective_column(dataset: pd.DataFrame, objective: str):
    return dataset.drop(objective, axis="columns")
