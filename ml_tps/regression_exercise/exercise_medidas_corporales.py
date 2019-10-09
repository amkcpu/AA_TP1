import pandas as pd
import os
from ml_tps.utils.dataframe_utils import divide_in_training_test_datasets

dir_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FILEPATH = f"{dir_path}/../regression_exercise/data/"
DATA_FILENAME = "data_multiple_regression_exercice.csv"
OBJECTIVE = "weight"
TRAIN_PCTG = 0.6


def main():
    dataset = pd.read_csv(DEFAULT_FILEPATH + DATA_FILENAME, sep=" ")
    train, test = divide_in_training_test_datasets(dataset=dataset, train_pctg=TRAIN_PCTG)

    # Multiple Regression using all variables


    # Other multiple regression dropping some variables following some criterium (forward/backward/mixed selection)


    # Compare both modelos using test set and calculate RSS for each case

    a = 1


if __name__ == '__main__':
    main()
