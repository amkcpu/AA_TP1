import numpy as np
import pandas as pd
import click
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_FILEPATH_DEFAULT = f"{dir_path}/data/britishPreference.xlsx"
LIKENESS_ARRAY_DEFAULT = "[1, 0, 1, 1, 0]"


def parse_int_list(string : str):
    strings = string.replace("[","").replace("]","").split(',')
    return [int(s) for s in strings]

@click.command(name="e1_2")
@click.option("--data-filepath", default=DATA_FILEPATH_DEFAULT)
@click.option("--likeness-array", default=LIKENESS_ARRAY_DEFAULT)
def main(data_filepath, likeness_array):
    dataset = pd.read_excel(data_filepath)      # import data set
    example_parameter = parse_int_list(likeness_array) # given by exercise specification
    given_example = pd.DataFrame(np.array(example_parameter).transpose())         # example representation

    # Calculate P(E) and P(I)
    no_of_examples, no_of_attributes = dataset.shape            # get data set size
    count_english, count_irish = dataset["Nacionalidad"].value_counts()         # number of English and Irish entries
    prob_english = count_english / no_of_examples
    prob_irish = count_irish / no_of_examples

    # Loop over all attributes associated --> P(a_i|v_j) = P(a_i AND v_j)/P(v_j)
    dataset_english = dataset[dataset["Nacionalidad"] == "E"]   # subset of English entries
    dataset_irish = dataset[dataset["Nacionalidad"] == "I"]     # subset of Irish entries
    params = pd.DataFrame(np.array(np.zeros((2, 5))))           # Instantiate parameter matrix

    # TODO Laplace smoothing?
    for i in range(0, no_of_attributes - 1):                   # for English attributes
        value_counts = dataset_english.iloc[:, i].value_counts(sort=False)    # returns each value found in column i and their frequency
        ones = value_counts[1]                                 # get number of 1s for this column
        params[i][0] = ones/count_english                      # divide by no. of English training examples to get frequency

    for j in range(0, no_of_attributes - 1):                   # for Irish attributes
        value_counts = dataset_irish.iloc[:, j].value_counts(sort=False)
        ones = value_counts[1]
        params[j][1] = ones/count_irish

    # Calculate hypothesis for English and Scottish for given example
    hypothesis = pd.DataFrame(np.array(np.ones((2, 1))))

    for i in given_example.index[given_example[0] == 1]:                        # only multiply params where example == 1
        hypothesis.iat[0, 0] *= params.iat[0, i]*given_example.iat[i, 0]        # calculate probability for English
        hypothesis.iat[1, 0] *= params.iat[1, i]*given_example.iat[i, 0]        # prob for Irish

    hypothesis.iat[0, 0] *= prob_english                                        # multiply with P(English)
    hypothesis.iat[1, 0] *= prob_irish                                          # multiply with P(Irish)

    # Output
    print("==================== Data ====================")
    print(dataset, "\n")

    print("P(English) =", prob_english)
    print("P(Irish) =", prob_irish, "\n")

    print("==================== Trained parameters ====================")
    params = params.rename(columns={0: "Scones", 1: "Cerveza", 2: "Whiskey", 3: "Avena", 4: "Futbol"}, index={0: "English", 1: "Irish"})
    print(params, "\n")

    print("==================== Hypothesis ====================")
    print("Given example: x = (", given_example.transpose().to_string(index=False, header=False), ")\n")

    hypothesis = hypothesis.rename(columns={0: "Probability"}, index={0: "English", 1: "Irish"})    # legible column and index names
    print(hypothesis, "\n")

    # Output and prediction in words
    if hypothesis.iat[0, 0] > hypothesis.iat[1, 0]:     # Predict English
        print("Because P(English|Example) =", hypothesis.iat[0, 0], "is bigger than P(Irish|Example) =", hypothesis.iat[1, 0], ", ")
        print("we predict the given example to be an English person.")
    elif hypothesis.iat[0, 0] < hypothesis.iat[1, 0]:   # Predict Irish
        print("Because P(Irish|Example) =", hypothesis.iat[0, 0], "is smaller than P(Irish|Example) =", hypothesis.iat[1, 0], ", ")
        print("we predict the given example to be an Irish person.")
    else:                                               # Same likelihood
        print("Same probability for each class.")


if __name__ == '__main__':
    main()