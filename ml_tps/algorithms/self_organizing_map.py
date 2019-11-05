import pandas as pd

# Self organizing Kohonen maps (SOM)
# Algorithm
# Initialize weights
# for vector in all_example_vectors or until certain number of iterations N:
#   Randomly choose weight vector
#   Search for most similar weights using euclidean distance -> choose best matching unit (BMU)
#   Update step: Update weights as well as the neighboring weights


class SelfOrganizingMap:

    def __init__(self):