# Trabajo Practico 3 - Ejercicio 3
# Picture segmentation

import datetime
import pandas as pd
import numpy as np
import os
from sklearn import svm
from ml_tps.utils.dataframe_utils import divide_in_training_test_datasets, scale_dataset, \
    seperateDatasetObjectiveData, get_test_train_X_y
import matplotlib.pyplot as plt
from ml_tps.utils.image_utils import read_image_to_dataframe
from ml_tps.tp3.svm_utils import test_svm_configurations

dir_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FILEPATH = f"{dir_path}/../tp3/data/"

def main():
    filepath_sky = DEFAULT_FILEPATH + "cielo.jpg"
    filepath_cow = DEFAULT_FILEPATH + "vaca.jpg"
    filepath_grass = DEFAULT_FILEPATH + "pasto.jpg"
    filepath_given_test_image = DEFAULT_FILEPATH + "cow.jpg"
    filepath_own_test_image = DEFAULT_FILEPATH + "field_sky_test_image.jpg"

    objective_sky = "Sky"
    objective_cow = "Cow"
    objective_grass = "Grass"

    # a)  Construir un conjunto de datos para entrenamiento, indicando para cada muestra a qué clase pertenece.
    data_sky = read_image_to_dataframe(filepath_sky)
    data_cow = read_image_to_dataframe(filepath_cow)
    data_grass = read_image_to_dataframe(filepath_grass)
    data_given_test_image = read_image_to_dataframe(filepath_given_test_image)
    data_own_test_image = read_image_to_dataframe(filepath_own_test_image)

    # Add objective columns
    data_sky["Objective"] = objective_sky
    data_cow["Objective"] = objective_cow
    data_grass["Objective"] = objective_grass

    # Scale image data
    data_sky = scale_dataset(dataset=data_sky, objective="Objective")
    data_cow = scale_dataset(dataset=data_cow, objective="Objective")
    data_grass = scale_dataset(dataset=data_grass, objective="Objective")
    data_given_test_image = scale_dataset(dataset=data_given_test_image, objective=None)
    data_own_test_image = scale_dataset(dataset=data_own_test_image, objective=None)

    # TODO Merge data for classification
    merged_data = 1

    # b)  Divide data set into training and cross-validation (cv) set
    X_train, y_train, X_cv_set, y_cv_set = get_test_train_X_y(merged_data, objective=objective_sky)

    # c)  Utilizar el método SVM para clasificar los pixels del conjunto de prueba, entrenando con el conjunto de entrenamiento.
    kernels = ["rbf", "poly", "linear", "sigmoid"]
    c_values = list(np.logspace(-3, 2, 6))

    # TODO test_svm_configurations() for each example
    svm_values, best_svm = test_svm_configurations(kernels=kernels, c_values=c_values, X_train=X_train, y_train=y_train,
                                                   X_cv_set=X_cv_set, y_cv_set=y_cv_set, alwaysPrintConfusionMatrix=True)

    # Confusion Matrix for each one (edit test_svm_configs()?)

    # d)  ¿Cuál es el núcleo que da mejores resultados? Pensar una justificación teórica para la respuesta.
    # TODO Average over each kernel's accuracy and get best

    # e)  Con el mismo método ya entrenado clasificar todos los pixels de la imagen.
    # TODO classify all pixels in raw_given_test_image

    # TODO Put all of this in method segment_image()
    # TODO Translate classification to color (3 columns representing RGB)
    #  -> Get relevant indices: predicts[predicts == "sky"].index etc.
    #  -> make DataFrame with np.zeros([len(predicts), 3])
    #  -> for indices fill respective color in column
    #  -> Pasto = Green: 0,255,0 | Vaca = Red: 255,0,0 | Cielo = Blue: 0,0,255
    # TODO convert DataFrame to nparray to be able to pass it to imshow()
    #  -> convert_dataframe_to_image()
    #  -> df.to_numpy()
    # TODO Reshape array
    #  -> predicts.reshape(raw_given_test_image.shape)
    # TODO Draw image
    #  -> plt.imshow() + plt.show()

    # f)  Con el mismo método ya entrenado clasificar todos los pixels de otra imagen.
    # TODO Same as in e) with raw_own_test_image

    a = 1

if __name__ == '__main__':
    main()