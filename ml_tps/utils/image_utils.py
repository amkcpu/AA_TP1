import pandas as pd
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt


def read_image_to_dataframe(filepath: str):
    raw_data = img.imread(filepath)
    return rgb_to_dataframe(raw_data)


def rgb_to_dataframe(raw_image_data):
    dataframe = raw_image_data.reshape(-1, 3)   #  Unroll data from 3d to 2D array with 3 columns (for R, G, B)
    dataframe = pd.DataFrame(dataframe)

    return dataframe


def dataframe_to_rgb(image_data: pd.DataFrame, height: int, width: int):
    rgb_image = image_data.to_numpy()
    rgb_image = rgb_image.reshape(height, width, 3)

    return rgb_image


# Segment image and give each region a different color
def segment_image(fitted_classifier, image: pd.DataFrame, classes: list, rgb_colors: list):
    predictions = pd.Series(fitted_classifier.predict(image))
    segmented_image = pd.DataFrame(np.zeros([len(predictions), 3]))

    i = 0
    for cls in classes:
        indices = predictions[predictions == cls].index.values  # Sky -> Blue (0,0,225)
        segmented_image.loc[indices, :] = rgb_colors[i]
        i += 1

    return segmented_image


def segment_and_draw(fitted_classifier, image: pd.DataFrame, classes: list, rgb_colors: list, height: int, width: int):
    segmented_image = segment_image(fitted_classifier=fitted_classifier, image=image,
                                    classes=classes, rgb_colors=rgb_colors)
    drawable_image = dataframe_to_rgb(image_data=segmented_image, height=height, width=width)

    plt.imshow(drawable_image)
    plt.show()
