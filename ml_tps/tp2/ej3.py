# Trabajo practico 2 - Ejercicio 3

import pandas as pd

DEFAULT_FILEPATH = "data/reviews_sentiment.csv"


def main():
    # TODO Cleanup missing values
    dataset = pd.read_csv(DEFAULT_FILEPATH, sep=';')  # review_sentiments.csv is semicolon-separated (;)
    no_rows, no_columns = dataset.shape

    # ========== a) Mean no. of words of reviews valued with 1 star
    one_star_ratings = dataset[dataset["Star Rating"] == 1]
    one_star_review_mean_words = sum(one_star_ratings["wordcount"]) / len(one_star_ratings)

    # ========== b) Divide data set into two parts, training and evaluation set


    # ========== c) Apply KNN and Weighted-distances KNN to predict review ratings (stars)
    # k = 5

    # ========== d) Calculate classifier precision and confusion matrix

    # ============== Final printout ==============
    print("========== Data set info ==========")
    print("Number of rows: ", no_rows)
    print("Number of columns: ", no_columns)

    print("\n========== Ejercicio a) ==========")
    print("Mean no. of words of 1-star-reviews:", one_star_review_mean_words)

    a = 1


if __name__ == '__main__':
    main()
