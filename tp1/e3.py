# Naive Bayes Text classifier

# Set categories: E. g. soccer, tennis, fighting (UFC, MMA), rugby, volleyball, hockey, basketball

# Select feature vectors (appearance of specific words)

# Calculate feature vector frequency P(feature_i|class_j) = (no. of times feature_i appears in class_j) / (total count of features in class_j)
    # remember to use Laplace (or other) smoothing

# Divide data set into training and validation set

# Classify validation data and get results
    # Calculate feature vectore frequency
    # Multiply P(feature_i|class_j) for all i, j, for each example
    # Choose class with highest likelihood for each example

# Construct confusion matrix

# Calculate accuracy, precision, true positives, false positives, F1-score
