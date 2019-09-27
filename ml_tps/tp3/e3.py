# Trabajo Practico 3 - Ejercicio 3
# Picture segmentation

import datetime
import pandas as pd
import numpy as np
import os
from sklearn import svm
from ml_tps.utils.evaluation_utils import getConfusionMatrix, computeAccuracy
from ml_tps.utils.dataframe_utils import divide_in_training_test_datasets, scale_dataset, seperateDatasetObjectiveData
import matplotlib.image as img
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FILEPATH = f"{dir_path}/../tp3/data/"

def main():
    # a)  Construir un conjunto de datos para entrenamiento, indicando para cada muestra a qué clase pertenece.
    pictures = []
    picture_names = ["cow", "cielo", "pasto", "vaca"]

    for pic in picture_names:
        pictures.append(img.imread(DEFAULT_FILEPATH + pic + ".jpg"))

    # Show all four pictures together
    i = 221
    for pic in pictures:
        plt.subplot(i)
        plt.imshow(pic)
        i += 1
    plt.show()

    # b)  Dividir aleatoriamente el conjunto de datos en dos conjuntos, uno de entrenamiento y uno de prueba.
    # c)  Utilizar el método SVM para clasificar los pixels del conjunto de prueba, entrenando con el conjunto de entrenamiento.
    #       Utilizar diferentes núcleos y diferentes valores del parámetro C. Construir la matriz de confusión para cada caso.
    # d)  ¿Cuál es el núcleo que da mejores resultados? Pensar una justificación teórica para la respuesta.
    # e)  Con el mismo método ya entrenado clasificar todos los pixels de la imagen.
    # f)  Con el mismo método ya entrenado clasificar todos los pixels de otra imagen.

    a = 1

if __name__ == '__main__':
    main()