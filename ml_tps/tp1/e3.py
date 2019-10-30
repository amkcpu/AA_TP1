# TP1 - E3: Naive Bayes Text classifier
import click
import pandas as pd
import os  # for looping over files in directory
import datetime  # measure runtime
from ml_tps.utils.text_processing import tf_idf, extract_words_from_text
from ml_tps.utils.evaluation import f1_score, getConfusionMatrix, computeAccuracy, \
    computePrecision, computeRecall

dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_FILEPATH_DEFAULT = f"{dir_path}/data/aa_bayes.tsv"
TRAINING_PERCENTAGE_DEFAULT = 0.02
VALIDATION_AMOUNT_DEFAULT = 300


@click.command("e1_3")
@click.option("--data-filepath", default=DATA_FILEPATH_DEFAULT)
@click.option("--training-percentage", default=TRAINING_PERCENTAGE_DEFAULT)
@click.option("--keyword-amount", default=20)
@click.option("--validation-amount", default=VALIDATION_AMOUNT_DEFAULT)
def main(data_filepath, training_percentage, keyword_amount, validation_amount):
    initial_time = datetime.datetime.now()

    # ============== Variable setup ==============
    path = data_filepath  # Set path containing text documents as .txt files
    no_of_keywords = keyword_amount  # how many highest scoring words on TF-IDF are selected as features
    no_of_validation_examples = validation_amount

    # ============== Get and process data ==============
    # Extract information and save in DataFrame
    objective = "categoria"
    predicted = "titular"

    data_set = pd.read_csv(path, sep="\t")
    data_set = data_set[data_set[objective] != "Noticias destacadas"]  # Leave out massive, unspecific "Noticias destacadas" category

    # Split data set into data subsets by category
    available_classes = pd.Series(data_set[objective].unique()).dropna().sort_values()
    categories = {}
    for cls in available_classes:
        categories[cls] = data_set[data_set[objective] == cls]

    # Extract words from each data (sub-)set
    # TODO Consider implementing Porter stemming to reduce redundancy
    #  http://www.3engine.net/wp/2015/02/stemming-con-python/
    words_then = datetime.datetime.now()  # for measuring runtime

    words = list()  # will contain words from all data subsets, each as one list element

    for category_name, category_data in categories.items():
        words_this_category = pd.DataFrame()
        counter = 0
        for row in category_data[predicted]:
            if counter >= int(len(category_data) * training_percentage):
                break
            words_one_title = extract_words_from_text(text=row, prevent_uppercase_duplicates=True)
            words_one_title.columns = [category_name + "_" + predicted + "_" + str(counter)]
            words_this_category = pd.concat([words_this_category, words_one_title], axis=1)
            counter += 1

        words.append(words_this_category)

    print("Runtime of word parsing:", divmod((datetime.datetime.now() - words_then).total_seconds(), 60), "\n")

    # ============== Compute TF-IDF scores and, based on those, choose keywords ==============
    then = datetime.datetime.now()  # perf measurement

    tf_idf_scores = list()
    for words_this_category in words:
        tf_idf_scores.append(tf_idf(words_this_category))  # word frequencies for Bayes classifier, contains NaN

    print("Runtime of TF-IDF:", divmod((datetime.datetime.now() - then).total_seconds(), 60))

    # Get x words with maximum TF-IDF scores for each category, with associated words as indices
    keywords = list()
    for scores_this_category in tf_idf_scores:
        keywords_this_category = scores_this_category.max(axis=1).sort_values(ascending=False).head(no_of_keywords)
        keywords.append(keywords_this_category)

    # ======= "Train" parameters: Retrieve frequency in respective category for each keyword =======
    keyword_frequency = list()

    for i, cat_dataset in enumerate(categories.values()):
        current_category_word_count = pd.DataFrame()

        for j in range(0, int(len(cat_dataset) * training_percentage)):
            counts_one_example = words[i].iloc[:, j].value_counts()  # get word count in one example (column)
            current_category_word_count = pd.concat([current_category_word_count, counts_one_example], axis=1,
                                                    sort=True)

        category_no_of_words = current_category_word_count.sum().sum()  # get overall number of words in this category
        temp = current_category_word_count.sum(axis=1) / category_no_of_words  # frequency of words in category
        temp = temp[temp.index.isin(keywords[i].index)]  # choose subset of words as keywords selected above
        keyword_frequency.append(temp)

    # ============== Bayes classifier ==============
    validation_examples = data_set.sample(n=no_of_validation_examples)  # random sample from data set
    validation_example_predictions = list()

    for i in range(0, no_of_validation_examples):  # get one example at a time
        example_words = extract_words_from_text(validation_examples[predicted].iat[i],
                                                True)  # get words for given example
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
                                1 / len(categories))  # P(P_i) = P(P_i|cat1)*P(cat1) + P(P_i|cat2)*P(cat2) + ...
                    except KeyError:
                        continue

                prob_this_category += prob_keyword_in_category * prob_keyword_in_entire_dataset  # P(Cat) = P(Cat|Key1)*P(Key1) + P(Cat|Key2)*P(Key2) + ...
            category_wise_prob.append(prob_this_category)

        predicted_class = category_wise_prob.index(max(category_wise_prob))  # find class with highest probability
        predicted_class_name = list(categories.keys())[predicted_class]  # find associated class name
        validation_example_predictions.append(predicted_class_name)

    # ============== Evaluation ==============
    predictions = pd.Series(validation_example_predictions)
    actual = validation_examples[objective]
    confusion_matrix = getConfusionMatrix(predictions, actual)

    # Eval metrics
    accuracy = computeAccuracy(predictions, actual)
    precision = computePrecision(predictions, actual)
    recall = computeRecall(predictions, actual)
    f1 = f1_score(precision, recall)

    # ============== Final printout ==============
    print("\n========== Data set info ==========")
    print("Number of entries in data set: ", data_set.shape[0], " Number of attributes: ", data_set.shape[1])
    print("Categories found:", categories.keys())

    print("\n========== Classifier info ==========")
    print("Number of training examples: ", current_category_word_count.shape[0], "x", len(categories),
          "=", current_category_word_count.shape[0] * len(categories))
    print("Number of validation examples: ", no_of_validation_examples)

    print("\n========== Evaluation metrics ==========")
    print("Confusion matrix:", confusion_matrix)
    metrics = pd.Series({"Accuracy:": accuracy,
                         "Precision": precision,
                         "Recall": recall,
                         "F1-score": f1})
    print(pd.DataFrame(metrics, columns=["Evaluation metrics"]))

    print("\nTotal runtime:", divmod((datetime.datetime.now() - initial_time).total_seconds(), 60))
    a = 1


if __name__ == "__main__":
    main()
