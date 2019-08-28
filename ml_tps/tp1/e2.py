import numpy as np
import pandas as pd
import click
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_FILEPATH_DEFAULT = f"{dir_path}/data/britishPreference.xlsx"
LIKENESS_ARRAY_DEFAULT = "[1, 0, 1, 1, 0]"
NATIONALITIES_LIST = "[I,E]"


def parse_int_list(string: str):
    strings = string.replace("[", "").replace("]", "").split(',')
    return [int(s) for s in strings]


def parse_string_list(string: str):
    return string.replace("[", "").replace("]", "").split(',')


@click.command(name="e1_2")
@click.option("--data-filepath", default=DATA_FILEPATH_DEFAULT)
@click.option("--likeness-array", default=LIKENESS_ARRAY_DEFAULT)
@click.option("--nationalities", default=NATIONALITIES_LIST)
def main(data_filepath, likeness_array, nationalities):
    dataset = pd.read_excel(data_filepath)      # import data set
    example_parameter = parse_int_list(likeness_array) # given by exercise specification
    given_example = np.array(example_parameter).transpose()         # example representation
    nationalities = parse_string_list(nationalities)

    # Loop over all attributes associated --> P(a_i|v_j) = P(a_i AND v_j)/P(v_j)
    nationalities_datasets = [
        dataset[
            dataset["Nacionalidad"] == n
            ] for n in nationalities
    ]

    # We want P(n|e) with n nationality, e example (e.g. 1, 0, 1, 1, 0)
    # P(n|e) = P(e|n) * P(n) / P(e)
    #        P(e_1,...,N|n) = π P(e_i|n)
    # P(e) = P(e_1,...,N)   = π P(e_i)

    # Calculate each P(n)
    no_of_examples, no_of_attributes = dataset.shape  # get data set size
    attributes = dataset.columns[:-1]                 # ignore last column of data set containing category (E, I)
    nat_count = np.array([
        len(d) for d in nationalities_datasets
    ])  # number of English and Irish entries
    nat_prob = nat_count / no_of_examples

    # Calculate each P(e_att|att)
    general_likeness = dataset.sum(axis=0)[:-1] / no_of_examples    # prob for each attribute (e. g. cerveza)
    likeness = np.array(np.zeros((len(nationalities), no_of_attributes - 1)))  # Instantiate parameter matrix

    for i, nat_dataset in enumerate(nationalities_datasets):
        likeness[i] = np.array(nat_dataset.sum(axis=0)[:-1])

    likeness = (likeness.T / nat_count).T

    p_e = np.prod(np.multiply(general_likeness, given_example) + np.multiply(1-general_likeness, 1-given_example))
    p_e_n = np.prod(np.multiply(likeness, given_example) + np.multiply(1-likeness, 1-given_example), axis=1)

    # For laplace smoothing do not divide, checks for 0, and sum 1 to every one, then divide by (nat count * 2)

    # Calculate hypothesis for English and Scottish for given example
    hypothesis = np.multiply(p_e_n, nat_prob) / p_e

    print("==================== Data ====================")
    print("### Dataset ###")
    print(dataset)

    print("\n### P(n) ###")
    for i, n in enumerate(nationalities):
        print(f"P({n}) = {nat_prob[i]}")

    print("\n==================== Trained parameters ====================")
    print("### P(e_i) ###")
    print(general_likeness)

    likeness = pd.DataFrame(likeness).rename(columns={key: value for (key, value) in enumerate(attributes)},
                                             index={key: value for (key, value) in enumerate(nationalities)})

    print("\n### P(e_i|n) ###")
    print(likeness)

    print("\n### P(e) ###")
    print(f"P(e): x = {p_e}")
    p_e_n = pd.DataFrame(p_e_n).rename(index={key: value for (key, value) in enumerate(nationalities)})

    print("\n### P(e|n) ###")
    print(f"P(e|n): x = {p_e_n}")

    print("\n==================== Hypothesis ====================")
    print(f"Given example: x = {given_example}")

    hypothesis = pd \
        .DataFrame(hypothesis.T)\
        .rename(columns={0: "Probability"},
                index={key: value for (key, value) in enumerate(nationalities)}) # legible column and index names
    print("\n", hypothesis)

    # Multiple nationalities
    # # Output and prediction in words
    # if hypothesis.iat[0, 0] > hypothesis.iat[1, 0]:     # Predict English
    #     print("Because P(English|Example) =", hypothesis.iat[0, 0], "is bigger than P(Irish|Example) =", hypothesis.iat[1, 0], ", ")
    #     print("we predict the given example to be an English person.")
    # elif hypothesis.iat[0, 0] < hypothesis.iat[1, 0]:   # Predict Irish
    #     print("Because P(Irish|Example) =", hypothesis.iat[0, 0], "is smaller than P(Irish|Example) =", hypothesis.iat[1, 0], ", ")
    #     print("we predict the given example to be an Irish person.")
    # else:                                               # Same likelihood
    #     print("Same probability for each class.")


if __name__ == '__main__':
    main()