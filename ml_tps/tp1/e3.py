# Naive Bayes Text classifier
import click
import numpy as np
import pandas as pd
import os  # for looping over files in directory
import datetime  # measure runtime
from ml_tps.utils.text_analysis_utils import tf_idf, extract_words_from_text
from ml_tps.utils.evaluation_utils import f1_score

dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_FILEPATH_DEFAULT = f"{dir_path}/data/aa_bayes.tsv"
TRAINING_PERCENTAGE_DEFAULT = 0.01
VALIDATION_AMOUNT_DEFAULT = 50

@click.command("e1_3")
@click.option("--data-filepath", default=DATA_FILEPATH_DEFAULT)
@click.option("--training-percentage", default=TRAINING_PERCENTAGE_DEFAULT)
@click.option("--training-amount", required=False, type=int)
@click.option("--keyword-amount", default=20)
@click.option("--validation-amount", default=VALIDATION_AMOUNT_DEFAULT)
def main(data_filepath, training_percentage, training_amount, keyword_amount, validation_amount):
    # ============== Variable setup ==============
    path = data_filepath  # Set path containing text documents as .txt files
    # no_of_training_examples = training_percentage * len(data_set)
    no_of_keywords = keyword_amount  # how many highest scoring words on TF-IDF are selected as features
    no_of_validation_examples = validation_amount

    # ============== Get and process data ==============
    # Extract information and save in DataFrame
    data_set = pd.read_csv(path, sep="\t")
    data_set = data_set[data_set["categoria"] != "Noticias destacadas"]  # Leave out massive, unspecific "Noticias destacadas" category

    # Split data set into data subsets by category
    # TODO find smarter way to separate given data sets
    data_set_deportes = data_set[data_set["categoria"] == "Deportes"]
    data_set_destacadas = data_set[data_set["categoria"] == "Destacadas"]
    data_set_economia = data_set[data_set["categoria"] == "Economia"]
    data_set_entretenimiento = data_set[data_set["categoria"] == "Entretenimiento"]
    data_set_internacional = data_set[data_set["categoria"] == "Internacional"]
    data_set_nacional = data_set[data_set["categoria"] == "Nacional"]
    data_set_salud = data_set[data_set["categoria"] == "Salud"]
    data_set_ciencia_tecnologia = data_set[data_set["categoria"] == "Ciencia y Tecnologia"]
    # data_set_dest = data_set[data_set["categoria"] == "Noticias destacadas"]  # Leave out massive, unspecific "Noticias destacadas" category

    categories = {"Deportes": data_set_deportes,
                  "Destacadas": data_set_destacadas,
                  "Economia": data_set_economia,
                  "Entretenimiento": data_set_entretenimiento,
                  "Internacional": data_set_internacional,
                  "Nacional": data_set_nacional,
                  "Salud": data_set_salud,
                  "Ciencia y Tecnologia": data_set_ciencia_tecnologia,
                  # "NoticiasDest": data_set_dest
                  }

    # Extract words from each data (sub-)set
    # TODO Consider implementing Porter stemming to reduce redundancy
    #  http://www.3engine.net/wp/2015/02/stemming-con-python/
    words_then = datetime.datetime.now()  # for measuring runtime

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

    # ============== Compute TF-IDF scores and, based on those, choose keywords ==============
    # TODO only get TF-IDF score for training set to reduce runtime
    then = datetime.datetime.now()  # perf measurement

    tf_idf_scores = list()

    for i in range(0, len(categories)):
        tfidf_result = tf_idf(words[i])  # word frequencies for Bayes classifier, contains NaN
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
            counts_one_example = words[i].iloc[:, j].value_counts()  # get word count in one example (column)
            current_category_word_count = pd.concat([current_category_word_count, counts_one_example], axis=1,
                                                    sort=True)

        category_no_of_words = current_category_word_count.sum().sum()  # get overall number of words in this category
        temp = current_category_word_count.sum(axis=1) / category_no_of_words  # frequency of words in category
        temp = temp[temp.index.isin(keywords[i].index)]  # choose subset of words as keywords selected above
        keyword_frequency.append(temp)

    # ============== Bayes classifier ==============
    validation_examples = data_set.sample(n=no_of_validation_examples)  # random sample from data set
    # TODO ensure that examples from each category are chosen, to solve problem described below (evaluation, confusion matrix)
    validation_example_predictions = list()

    for i in range(0, no_of_validation_examples):  # get one example at a time
        example_words = extract_words_from_text(validation_examples.iloc[i, 1], True)  # get words for given example
        category_wise_prob = list()

        for j in range(0, len(categories)):
            prob_this_category = 0
            prob_keyword_in_entire_dataset = 0

            for word in example_words.iterrows():
                try:  # TODO maybe it's necessary to smoothen the results here (Laplace smoothing)
                    prob_keyword_in_category = keywords[j][word[1].iat[0]]
                except KeyError:  # when word not found in list of trained keywords
                    continue

                for k in range(0, len(categories)):  # get entire data set probability for this word
                    try:
                        prob_keyword_in_entire_dataset += keywords[k][word[1].iat[0]] * (
                                    1 / 8)  # P(P_i) = P(P_i|cat1)*P(cat1) + P(P_i|cat2)*P(cat2) + ...
                    except KeyError:
                        continue

                prob_this_category += prob_keyword_in_category * prob_keyword_in_entire_dataset  # P(Cat) = P(Cat|Key1)*P(Key1) + P(Cat|Key2)*P(Key2) + ...
            category_wise_prob.append(prob_this_category)

        predicted_class = category_wise_prob.index(max(category_wise_prob))  # find class with highest probability
        predicted_class_name = list(categories.keys())[predicted_class]  # find associated class name
        validation_example_predictions.append(predicted_class_name)

    validation_example_predictions_wrapper = pd.Series(validation_example_predictions)

    # ============== Evaluation ==============
    # confusion matrix
    # TODO this actually contains an error occurring in small validation samples,
    #  whereby some categories might not be predicted and the confusion matrix might
    #  not by square. An error gets thrown when initializing confusion_matrix_diag, but
    #  already the confusion_matrix itself is faulty.
    #  Using a large sample, this problem is avoided.
    validation_examples_actual_category = validation_examples["categoria"]
    validation_examples_actual_category.index = validation_example_predictions_wrapper.index
    confusion_matrix = pd.crosstab(validation_example_predictions_wrapper, validation_examples_actual_category,
                                   rownames=['Actual'], colnames=['Predicted'])
    confusion_matrix_diag = pd.Series(np.diag(confusion_matrix), index=confusion_matrix.index)  # get diagonal

    # TP rate, FP rate
    true_positive_rate = 0
    false_positive_rate = 0

    for i in range(0, len(confusion_matrix_diag)):
        column_sum = confusion_matrix.iloc[:, i].sum()      # amount of examples actually in category
        classified_correctly = confusion_matrix_diag[i]
        classified_incorrectly = column_sum - classified_correctly

        true_positive_rate += classified_correctly / column_sum
        false_positive_rate += classified_incorrectly / column_sum

    true_positive_rate /= len(confusion_matrix_diag)    # for average
    false_positive_rate /= len(confusion_matrix_diag)   # for average

    # accuracy
    accuracy = confusion_matrix_diag.sum() / confusion_matrix.sum().sum()

    # precision & recall
    recall = true_positive_rate
    precision = 0

    for i in range(0, len(confusion_matrix_diag)):
        precision += confusion_matrix_diag[i] / confusion_matrix.iloc[:, i].sum()

    precision /= len(confusion_matrix_diag)     # for average

    # compute f1 score
    f1 = f1_score(precision, recall)

    # ============== Final printout ==============
    print("\n========== Data set info ==========")
    print("Number of entries in data set: ", data_set.shape[0], " Number of attributes: ", data_set.shape[1])
    print("Categories found:", categories.keys())

    print("\n========== Classifier info ==========")
    print("Number of training examples: ", current_category_word_count.shape[0], "x", len(categories),
          "=", current_category_word_count.shape[0]*len(categories))
    print("Number of validation examples: ", no_of_validation_examples)

    print("\n========== Evaluation metrics ==========")
    print("Accuracy: ", accuracy, "\n")
    print("Confusion matrix:", confusion_matrix)
    print("\nTrue positive rate (TP): ", true_positive_rate)
    print("False positive rate (FP): ", false_positive_rate)
    print("Precision: ", precision)
    print("Recall (= true positive rate): ", recall)
    print("F1-score: ", f1)

    a = 1

if __name__ == "__main__":
    main()
