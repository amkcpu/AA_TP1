import pandas as pd
import os
from ml_tps.utils.dataframe_utils import get_test_train_X_y
from ml_tps.utils.regression_utils.linear_regression import LinearRegression

dir_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FILEPATH = f"{dir_path}/../regression_exercise/data/"
DATA_FILENAME = "data_multiple_regression_exercice.csv"
OBJECTIVE = "weight"
TRAIN_PCTG = 0.6

ALL_COLUMNS = ['biacromial', 'pelvic.breadth', 'bitrochanteric', 'chest.depth',
               'chest.diam', 'elbow.diam', 'wrist.diam', 'knee.diam', 'ankle.diam',
               'shoulder.girth', 'chest.girth', 'waist.girth', 'navel.girth', 'hip.girth',
               'thigh.girth', 'bicep.girth', 'forearm.girth', 'knee.girth', 'calf.girth',
               'ankle.girth', 'wrist.girth', 'age', 'height']


class Corporal(LinearRegression):

    def __init__(self, initial_b: pd.Series = None, column = "ALL"):
        super().__init__(initial_b)
        self.rss = 0
        self.column = column

    def fit(self, X: pd.DataFrame, y: pd.Series):
        super().fit(X, y)
        _, _, self.rss = self.calculate_sums_of_squares(X, y)


def filter_only(dataset: pd.DataFrame, columns=None):
    if columns is None:
        columns = ALL_COLUMNS
    return dataset.filter(items=columns + [OBJECTIVE])


def main(TOP):
    dataset = pd.read_csv(DEFAULT_FILEPATH + DATA_FILENAME, sep=" ")
    X_train, y_train, X_test, y_test = get_test_train_X_y(data=dataset, objective=OBJECTIVE, train_pctg=TRAIN_PCTG)

    # Multiple Regression using all variables

    full_classifier = Corporal()
    full_classifier.fit(X=X_train, y=y_train)

    # Other multiple regression dropping some variables following some criterium (forward/backward/mixed selection)

    classifiers = []

    columns = []
    testing_columns = ALL_COLUMNS.copy()
    for _ in range(TOP):
        for column in testing_columns:
            df = filter_only(X_train, columns + [column])
            lr = Corporal(column=column)
            lr.fit(df, y_train)
            classifiers.append(lr)

        classifiers_sorted = sorted(classifiers, key=lambda l: l.rss)
        columns.append(classifiers_sorted[0].column)
        testing_columns.remove(classifiers_sorted[0].column)

    df = filter_only(X_train, columns)
    filtered_classifier = Corporal(column="FILTERED")
    filtered_classifier.fit(X=df, y=y_train)

    # Compare both models using test set and calculate RSS for each case

    full_classifier.fit(X=X_test, y=y_test)
    filtered_classifier.fit(X=filter_only(X_test,columns), y=y_test)
    print(f"FULL RSS = {full_classifier.rss}")
    print(f"{TOP} VARIABLES RSS = {filtered_classifier.rss}")
    print(f"FULL R2-adj = {full_classifier.calculate_adjusted_r2(X_test,y_test)}")
    print(f"{TOP} VARIABLES R2-adj = {filtered_classifier.calculate_adjusted_r2(filter_only(X_test,columns),y_test)}")
    print(columns)


if __name__ == '__main__':
    main(5)
