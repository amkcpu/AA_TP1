import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_all_axes(data: pd.DataFrame, predictions: pd.Series, additional_points: pd.DataFrame = None) -> None:
    """Plots data along all available dimensions colored according to their predictions.

    Uses subplots with 4 columns and the necessary number of rows by default.

    Uses as Matplotlib's "Set3" as colormap, providing for 12 distinct class colors.

    :param data: Data set containing examples in rows, attributes in columns. Column names are used as axis labels.
    :param predictions: Class label for each example.
    :param additional_points: If in each plot additional points are to be plotted (like centroids in KMeans). Have to have the same column names as the rest of the data.
    """
    columns_list = list(data.columns)
    no_attributes = len(columns_list)

    # If we plot all columns against each other (without repetition), how many subplots to we need
    # Uses Gaussian addition formula n(n+1)/2, subtracting once the number of attributes, so there is no repetition
    no_subplots = int(((no_attributes * (no_attributes + 1)) / 2) - no_attributes)

    if no_attributes == 2:
        subplot_no_columns = 1
    elif no_attributes == 3:
        subplot_no_columns = 3
    else:
        subplot_no_columns = 4  # default is 4 for better legibility of the plots
    subplot_no_rows = int(np.ceil(no_subplots / subplot_no_columns))

    i = 1
    others = columns_list.copy()
    for col in columns_list:
        others.remove(col)
        for other in others:
            plt.subplot(subplot_no_rows, subplot_no_columns, i)
            plt.subplots_adjust(bottom=0.7)
            plt.scatter(data[other], data[col], c=predictions, s=10, cmap="Set3")
            plt.xlabel(other, fontweight="bold")
            plt.ylabel(col, fontweight="bold")
            if additional_points is not None:
                plt.scatter(additional_points[other], additional_points[col], c="black", marker="x", s=20)
            i += 1

    plt.show()
