import pandas as pd
import matplotlib.image as img


def read_image_to_dataframe(filepath: str):
    raw_data = img.imread(filepath)
    return rgb_to_dataframe(raw_data)


def rgb_to_dataframe(raw_image_data):
    dataframe = raw_image_data.reshape(-1, 3)   #  Unroll data from 3d to 2D array with 3 columns (for R, G, B)
    dataframe = pd.DataFrame(dataframe)

    return dataframe
