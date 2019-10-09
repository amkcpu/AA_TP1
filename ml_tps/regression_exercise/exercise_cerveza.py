import pandas as pd
import numpy as np
import os
from ml_tps.utils.dataframe_utils import get_test_train_X_y
from ml_tps.utils.regression_utils.linear_regression import fit_using_normal_equation, add_bias_to_dataset

dir_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FILEPATH = f"{dir_path}/../regression_exercise/data/"
DATA_FILENAME = "cervezas.csv"
OBJECTIVE = "Tiempo"
TRAIN_PCTG = 0.6


def main():
    dataset = pd.read_csv(DEFAULT_FILEPATH + DATA_FILENAME, sep=";")
    X_train, y_train, X_test, y_test = get_test_train_X_y(data=dataset, objective=OBJECTIVE, train_pctg=TRAIN_PCTG)

    # Predict according to "Cajas"
    X_train_cajas = X_train.drop("Distancia", axis=1)
    b_cajas = fit_using_normal_equation(X=X_train_cajas, y=y_train, plot=True)


    # Predict according to "Distancia"
    X_train_distancia = X_train.drop("Cajas", axis=1)
    b_distancia = fit_using_normal_equation(X=X_train_distancia, y=y_train, plot=True)

    # Compare models calculating RSS

    a = 1


if __name__ == '__main__':
    main()
