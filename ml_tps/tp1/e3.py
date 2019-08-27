# Naive Bayes Text classifier
import click
import numpy as np
import pandas as pd
import os  # for looping over files in directory
import datetime  # measure runtime
from ml_tps.utils.text_analysis_utils import tf_idf, extract_words_from_text

dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_FILEPATH_DEFAULT = f"{dir_path}/data/aa_bayes.tsv"
TRAINING_PERCENTAGE_DEFAULT = 0.01

@click.command("e1_3")
@click.option("--data-filepath", default=DATA_FILEPATH_DEFAULT)
@click.option("--training-percentage", default=TRAINING_PERCENTAGE_DEFAULT)
@click.option("--training-amount", required=False, type=int)
@click.option("--keyword-amount", default=20)
def main(data_filepath, training_percentage, training_amount, keyword_amount):
    # Set categories: E. g. soccer, tennis, fighting (UFC, MMA), rugby, volleyball, hockey, basketball
    example_labels = np.array(["soccer", "fighting", "tennis", "soccer", "soccer",
                               "rugby", "rugby", "volleyball", "tennis", "tennis",
                               "tennis", "hockey", "hockey", "hockey", "soccer"
                                                                       "soccer", "soccer", "soccer", "basketball",
                               "fighting"])  # labels are ordered!

    # Adjust important variables
    path = data_filepath  # Set path containing text documents as .txt files

    # Extract information and save in DataFrame
    data_set = pd.read_csv(path, sep="\t")
    data_set = data_set[data_set["categoria"]!="Noticias destacadas"]   # Leave out massive, unspecific "Noticias destacadas" category

    # More variables
    #no_of_training_examples = training_percentage * len(data_set)
    no_of_keywords = keyword_amount  # how many highest scoring words on TF-IDF are selected as features
    no_of_validation_examples = 20

    # TODO find smarter way to separate given data sets
    # data_set_dest = data_set[data_set["categoria"] == "Noticias destacadas"]  # Leave out massive, unspecific "Noticias destacadas" category
    data_set_deportes = data_set[data_set["categoria"]=="Deportes"]
    data_set_destacadas = data_set[data_set["categoria"]=="Destacadas"]
    data_set_economia = data_set[data_set["categoria"]=="Economia"]
    data_set_entretenimiento = data_set[data_set["categoria"]=="Entretenimiento"]
    data_set_internacional = data_set[data_set["categoria"]=="Internacional"]
    data_set_nacional = data_set[data_set["categoria"]=="Nacional"]
    data_set_salud = data_set[data_set["categoria"]=="Salud"]
    data_set_ciencia_tecnologia = data_set[data_set["categoria"]=="Ciencia y Tecnologia"]

    categories = {"Deporte": data_set_deportes, "Destacadas": data_set_destacadas, "Economia": data_set_economia,
                  "Entretenimiento": data_set_entretenimiento, "Internacional": data_set_internacional,
                  "Nacional": data_set_nacional, "Salud": data_set_salud,
                  "CienciaTec": data_set_ciencia_tecnologia,
                  #"NoticiasDest": data_set_dest
                  }

    # Get words
    # TODO Consider implementing Porter stemming to reduce redundancy
    #  http://www.3engine.net/wp/2015/02/stemming-con-python/
    words_then = datetime.datetime.now()    # for measuring runtime

    words = list()  # will contain words from all data subsets, each as one list element

    for category_name, category_data in categories.items():
        words_this_category = pd.DataFrame()

        for i in range(0, training_amount or int(len(category_data) * training_percentage)):
            words_one_title = extract_words_from_text(category_data.iat[i, 1], True)
            words_one_title.columns = [(category_name, i)]
            words_this_category = pd.concat([words_this_category, words_one_title], axis=1)

        words.append(words_this_category)

    words_now = datetime.datetime.now()
    print("Runtime of word parsing: ", divmod((words_now - words_then).total_seconds(), 60), "\n")

    # Get TF-IDF
    then = datetime.datetime.now()  # perf measurement

    tf_idf_scores = list()

    for i in range(0, len(categories)):
        tfidf_result = tf_idf(words[i])     # word frequencies for Bayes classifier, contains NaN
        tf_idf_scores.append(tfidf_result)

    now = datetime.datetime.now()
    print("Runtime of TF-IDF: ", divmod((now - then).total_seconds(), 60))

    # Get sorted maximum TF-IDF scores for each category, with associated words as indices
    max_tfidf_scores = list()

    for i in range(0, len(categories)):
        temp = tf_idf_scores[i].max(axis=1).sort_values(ascending=False)
        max_tfidf_scores.append(temp)

    # Select x highest scoring words for each category -> feature vectors
    keywords = list()

    for i in range(0, len(categories)):
        temp = max_tfidf_scores[i].head(no_of_keywords)
        keywords.append(temp)

    # Retrieve frequency in respective category for each selected word -> parameter
    keyword_frequency = list()

    for i, cat_dataset in enumerate(categories.values()):
        current_category_word_count = pd.DataFrame()

        for j in range(0, training_amount or int(len(cat_dataset) * training_percentage)):
            counts_one_example = words[i].iloc[:, j].value_counts() # get word count in one example (column)
            current_category_word_count = pd.concat([current_category_word_count, counts_one_example], axis=1, sort=True)

        category_no_of_words = current_category_word_count.sum().sum()  # get overall number of words in this category
        temp = current_category_word_count.sum(axis=1) / category_no_of_words   # frequency of words in category
        temp = temp[temp.index.isin(keywords[i].index)]        # choose subset of words as keywords selected above
        keyword_frequency.append(temp)

    # Bayes classifier
    validation_examples = data_set.sample(n=no_of_validation_examples)  # random sample from data set
    validation_example_predictions = list()

    for i in range(0, no_of_validation_examples):
        example_words = extract_words_from_text(validation_examples.iloc[i, 1], True)     # get one example at a time
        category_wise_prob = list()

        for j in range(0, len(categories)):
            prob_this_category = 0
            for word in example_words.iterrows():
                try:        # TODO maybe it's necessary to smoothen the results here (Laplace smoothing)
                    prob_keyword_in_category = keywords[j][word[1].iat[0]]
                except KeyError:
                    continue

                prob_keyword_in_entire_dataset = 0

                for k in range(0, len(categories)):     # get entire dataset probability for this word
                    try:
                        prob_keyword_in_entire_dataset += keywords[k][word[1].iat[0]] * (1/8)    # P(P_i) = P(P_i|cat1)*P(cat1) + P(P_i|cat2)*P(cat2)
                    except KeyError:
                        continue

                prob_this_category *= prob_keyword_in_category * prob_keyword_in_entire_dataset # P(Cat) = P(Cat|Key1)*P(Key1) + P(Cat|Key2)*P(Key2)
            category_wise_prob.append(prob_this_category)

        predicted_class = category_wise_prob.index(max(category_wise_prob))
        predicted_class_name = list(categories.keys())[predicted_class]
        validation_example_predictions.append(predicted_class_name)

        # example.contains(list_of_keywords)
        # for keyword found: multiply keyword.frequency
        # at the end multiply with p(given_category)
        # find class with highest probability -> prediction

    # evaluation
        # confusion matrix
        # TP rate, FP rate
        # accuracy, precision, recall, F1-score



    a = 1
    # ============ Old instructions ==============
    # Calculate feature vector frequency P(feature_i|class_j) = (no. of times feature_i appears in class_j) / (total count of features in class_j)
    # remember to use Laplace (or other) smoothing

    # Divide data set into training and validation set

    # Classify validation data and get results
    # Calculate feature vector frequency
    # Multiply P(feature_i|class_j) for all i, j, for each example

if __name__ ==  "__main__":
    main()