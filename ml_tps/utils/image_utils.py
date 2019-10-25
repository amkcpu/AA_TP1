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


def dataframe_to_colored_image(image_data: pd.DataFrame, height: int, width: int):
    rgb_image = image_data.to_numpy()
    rgb_image = rgb_image.reshape(height, width, 3)

    return rgb_image


# TODO Merge classes and rgb_colors params into one colors_per_class param (dict)
# TODO Optimize for-loop
def segment_and_draw_image(fitted_classifier, image: pd.DataFrame, classes: list, rgb_colors: list, height: int, width: int):
    """Segment image by class using given classifier, give each region a different color and draw final image using Pyplot.

    :param fitted_classifier:   Classifier to use for image segmentation.
    :param image:               Image as unrolled pandas.DataFrame (2-dimensional).
    :param classes:             List of classes that classifier can predict.
    :param rgb_colors:          List of RGB colors with one color corresponding to each class. Each list element is a
                                    (value_red, value_green, value_blue) tuple with each value in range from 0 to 255.
    :param height:              Height of final image for appropriate re-enrolling.
    :param width:               Width of final image for appropriate re-enrolling.
    """
    predictions = pd.Series(fitted_classifier.predict(image))
    segmented_image = pd.DataFrame(np.zeros([len(predictions), 3]))

    i = 0
    for cls in classes:
        indices = predictions[predictions == cls].index.values  # Sky -> Blue (0,0,225)
        segmented_image.loc[indices, :] = rgb_colors[i]
        i += 1

    drawable_image = dataframe_to_colored_image(image_data=segmented_image, height=height, width=width)

    plt.imshow(drawable_image)
    plt.show()
