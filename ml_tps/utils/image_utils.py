import pandas as pd
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt


def read_image_to_dataframe(filepath: str):
    raw_data = img.imread(filepath)
    return colored_image_to_dataframe(raw_data)


def colored_image_to_dataframe(raw_image_data):
    """Unroll data from 3D colored image to 2D DataFrame with 3 columns (for R, G, B)."""
    dataframe = raw_image_data.reshape(-1, 3)
    dataframe = pd.DataFrame(dataframe)

    return dataframe


def dataframe_to_colored_image(image_data: pd.DataFrame, height: int, width: int, draw_image: bool):
    rgb_image = image_data.to_numpy()
    rgb_image = rgb_image.reshape(height, width, 3)

    if draw_image:
        plt.imshow(rgb_image)
        plt.show()

    return rgb_image


def segment_and_draw_image(fitted_classifier, image: pd.DataFrame, color_per_class: dict, height: int, width: int):
    """Segment image by class using given classifier, give each region a different color and draw final image.

    :param fitted_classifier:   Classifier to use for image segmentation.
    :param image:               Image as unrolled pandas.DataFrame (2-dimensional).
    :param color_per_class:     Dict where each key is a class that has an associated RGB color as value.
                                    RGB colors are [value_red, value_green, value_blue] lists
                                    with each value between 0 and 255.
    :param height:              Height of final image for appropriate re-enrolling.
    :param width:               Width of final image for appropriate re-enrolling.
    """
    predictions = pd.Series(fitted_classifier.predict(image))
    segmented_image = pd.DataFrame(np.zeros([len(predictions), 3]))

    for cls, color in color_per_class.items():
        indices = predictions[predictions == cls].index.values
        segmented_image.loc[indices, :] = color

    dataframe_to_colored_image(image_data=segmented_image, height=height, width=width, draw_image=True)
