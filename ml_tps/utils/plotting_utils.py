import pandas as pd
import matplotlib.pyplot as plt


def plot_all_axes(data: pd.DataFrame, predictions: pd.Series, additional_points: pd.DataFrame = None) -> None:
    """Plots data along all available dimensions, colored according to their predictions.

    Uses as Matplotlib's "Set3" as colormap, providing for 12 distinct class colors.

    :param data: Data set containing examples in rows, attributes in columns. Column names are used as axis labels.
    :param predictions: Class label for each example.
    :param additional_points: If in each plot additional points are to be plotted (like centroids in KMeans). Have to have the same column names as the rest of the data.
    """
    columns_list = list(data.columns)
    no_attributes = len(columns_list)
    no_columns = no_attributes - 1
    no_rows = no_attributes

    fig, ax = plt.subplots(no_rows, no_columns, sharey=True)
    fig.tight_layout()
    i = 0
    for col in columns_list:
        others = columns_list.copy()
        others.remove(col)
        j = 0
        ax[i, j].set_ylabel(col, fontweight="bold")
        for row in others:
            ax[i, j].scatter(data[row], data[col], c=predictions, s=10, cmap="Set3")
            if additional_points is not None:
                ax[i, j].scatter(additional_points[row], additional_points[col], c="black", marker="x", s=20)
            ax[i, j].set_xlabel(row)
            j += 1
        i += 1

    plt.show()
