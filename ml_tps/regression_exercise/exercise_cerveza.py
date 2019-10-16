import pandas as pd
import os
from ml_tps.utils.dataframe_utils import get_test_train_X_y
from ml_tps.utils.regression_utils.linear_regression import LinearRegression
import pandas_profiling


dir_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FILEPATH = f"{dir_path}/../regression_exercise/data/"
DATA_FILENAME = "cervezas.csv"
OBJECTIVE = "Tiempo"
TRAIN_PCTG = 0.6


def main():
    dataset = pd.read_csv(DEFAULT_FILEPATH + DATA_FILENAME, sep=";")
    X_train, y_train, X_test, y_test = get_test_train_X_y(data=dataset, objective=OBJECTIVE, train_pctg=TRAIN_PCTG)

    # Predict according to "Cajas"
    X_train_cajas = X_train.drop(columns=["Distancia"])
    cajas_classifier = LinearRegression()
    cajas_classifier.fit(X=X_train_cajas, y=y_train)
    cajas_classifier.plot(X=X_train_cajas, y=y_train)

    # Predict according to "Distancia"
    X_train_distancia = X_train.drop(columns=["Cajas"])
    distancia_classifier = LinearRegression()
    distancia_classifier.fit(X=X_train_distancia, y=y_train)
    distancia_classifier.plot(X=X_train_distancia, y=y_train)

    # Multiple regression using both
    multiple_classifier = LinearRegression()
    multiple_classifier.fit(X_train, y_train)
    # multiple_classifier.plot(X_train, y_train)

    # Compare models calculating RSS
    cajas_R2 = cajas_classifier.calculate_r2(X_train_cajas, y_train)
    cajas_adj_R2 = cajas_classifier.calculate_adjusted_r2(X_train_cajas, y_train)

    distancia_R2 = distancia_classifier.calculate_r2(X_train_distancia, y_train)
    distancia_adj_R2 = distancia_classifier.calculate_adjusted_r2(X_train_distancia, y_train)

    a = 1


if __name__ == '__main__':
    main()
