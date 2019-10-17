# Trabajo Practico 3 - Ejercicio 3
# Picture segmentation

import pandas as pd
import numpy as np
import os
from ml_tps.utils.dataframe_utils import divide_in_training_test_datasets, scale_dataset, get_test_train_X_y
from ml_tps.utils.image_utils import read_image_to_dataframe, segment_and_draw
from ml_tps.utils.svm_utils import test_svm_configurations

dir_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FILEPATH = f"{dir_path}/../tp3/data/"
PCTG_DATA_TO_USE = 0.1      # so that computation does not take too long, use smaller percentage of data


def main():
    filepath_sky = DEFAULT_FILEPATH + "cielo.jpg"
    filepath_cow = DEFAULT_FILEPATH + "vaca.jpg"
    filepath_grass = DEFAULT_FILEPATH + "pasto.jpg"
    filepath_given_test_image = DEFAULT_FILEPATH + "cow.jpg"
    filepath_own_test_image = DEFAULT_FILEPATH + "field_sky_test_image.jpg"

    sky_description = "Sky"
    cow_description = "Cow"
    grass_description = "Grass"
    objective_desc = "Objective"

    # a)  Construir un conjunto de datos para entrenamiento, indicando para cada muestra a qué clase pertenece.
    data_sky, discard_sky = divide_in_training_test_datasets(read_image_to_dataframe(filepath_sky), PCTG_DATA_TO_USE)
    data_cow, discard_cow = divide_in_training_test_datasets(read_image_to_dataframe(filepath_cow), PCTG_DATA_TO_USE)
    data_grass, discard_grass = divide_in_training_test_datasets(read_image_to_dataframe(filepath_grass), PCTG_DATA_TO_USE)
    data_given_test_image = read_image_to_dataframe(filepath_given_test_image)
    data_own_test_image = read_image_to_dataframe(filepath_own_test_image)

    # Add objective columns
    data_sky[objective_desc] = sky_description
    data_cow[objective_desc] = cow_description
    data_grass[objective_desc] = grass_description

    # Scale image data
    data_sky = scale_dataset(dataset=data_sky, objective=objective_desc, scaling_type="minmax")
    data_cow = scale_dataset(dataset=data_cow, objective=objective_desc, scaling_type="minmax")
    data_grass = scale_dataset(dataset=data_grass, objective=objective_desc, scaling_type="minmax")
    data_given_test_image = scale_dataset(dataset=data_given_test_image, objective=None, scaling_type="minmax")
    data_own_test_image = scale_dataset(dataset=data_own_test_image, objective=None, scaling_type="minmax")

    # Merge data for classification
    merged_data = pd.concat([data_sky, data_cow, data_grass], ignore_index=True)

    # b)  Divide data set into training and cross-validation (cv) set
    X_train, y_train, X_cv_set, y_cv_set = get_test_train_X_y(merged_data, objective=objective_desc)

    # c)  Utilizar el método SVM para clasificar los pixels del conjunto de prueba, entrenando con el conjunto de entrenamiento.
    kernels = ["rbf", "poly", "linear", "sigmoid"]
    c_values = list(np.logspace(-3, 2, 6))

    # test_svm_configurations() for classification + Confusion Matrix for each one
    svm_values, best_svm = test_svm_configurations(kernels=kernels, c_values=c_values, X_train=X_train, y_train=y_train,
                                                   X_cv_set=X_cv_set, y_cv_set=y_cv_set, printConfusionMatrices=True)
    print(svm_values)

    # d)  ¿Cuál es el núcleo que da mejores resultados? Pensar una justificación teórica para la respuesta.
    # Average over each kernel's accuracy and get best result
    kernels_average = pd.Series()
    i = 0
    for kernel in kernels:
        kernels_average[kernel] = svm_values[svm_values["Kernel"] == kernel]["CV set accuracy"].mean()
        i += 1

    best_kernel = kernels_average.sort_values(ascending=False).head(1)
    print(best_kernel)

    # e)  Con el mismo método ya entrenado clasificar todos los pixels de la imagen.
    best_svm.fit(X_train, y_train)

    red = [255, 0, 0]
    green = [0, 255, 0]
    blue = [0, 0, 255]
    colors = [blue, red, green]
    classes = [sky_description, cow_description, grass_description]

    segment_and_draw(fitted_classifier=best_svm, image=data_given_test_image,
                     classes=classes, rgb_colors=colors, height=760, width=1140)

    # f)  Con el mismo método ya entrenado clasificar todos los pixels de otra imagen
    segment_and_draw(fitted_classifier=best_svm, image=data_own_test_image,
                     classes=classes, rgb_colors=colors, height=627, width=1200)

    a = 1


if __name__ == '__main__':
    main()
