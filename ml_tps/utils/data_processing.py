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


def scale_dataset(dataset: pd.DataFrame, scaling_type: str = "standard", objective: str = None):
    """Scales/normalizes a given dataset.

    :param dataset:         Data set to be scaled.
    :param scaling_type:    Scaling type used to normalize. Supports "minmax", "maxabs", "robust", "standard".
    :param objective:       If passed, the column with this name will not be scaled.
    :returns: Scaled data set as pandas.DataFrame.
    """
    scaling_types = {"Minmax": ["minmax"],
                     "MaxAbs": ["maxabs"],
                     "Robust": ["robust"],
                     "Standard": ["standard"]}
    if scaling_type in scaling_types["Minmax"]:
        scaler = MinMaxScaler()  # (X - X.min()) / (X.max() - X.min())
    elif scaling_type in scaling_types["MaxAbs"]:
        scaler = MaxAbsScaler()  # X / abs(X.max())
    elif scaling_type in scaling_types["Robust"]:
        scaler = RobustScaler(quantile_range=(25.0, 75.0))  # X / X.quantile_range()
    elif scaling_type in scaling_types["Standard"]:
        scaler = StandardScaler(with_mean=True, with_std=True)  # (X - X.mean()) / X.std_deviation
    else:
        raise ValueError('"{0}" is not a supported scaling type. '
                         'The following dictionary lists the supported types as keys, '
                         'and the corresponding keywords as values: {1}.'.format(scaling_type, scaling_types))

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


def add_bias_to_dataset(dataset: pd.DataFrame, reset_columns: bool = False):
    """Add bias column consisting only of 1's to given data set as first column.

    :param dataset: Matrix to have bias column added
    :param reset_columns: If True, overrides columns with range from 0 to length (can be useful for preventing
                              matrix mismatches in matrix multiplication).
    :return: pandas.DataFrame with properties as noted in method description.
    """
    ones = pd.Series(np.ones(max(dataset.index) + 1))
    dataset_copy = dataset.copy()
    dataset_copy.insert(0, "Bias", ones)  # works inplace

    if reset_columns:
        dataset_copy.columns = range(0, len(dataset_copy.columns))

    return dataset_copy


def drop_objective_column(dataset: pd.DataFrame, objective: str):
    return dataset.drop(objective, axis="columns")
