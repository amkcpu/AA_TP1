import click
import numpy as np
import pandas as pd

from ml_tps.utils.probability_utils import control_probability

YOUNG_VALUES_DEFAULT = "0.95, 0.05, 0.02, 0.20"   # P(P_i|J)
OLD_VALUES_DEFAULT = "0.03, 0.82, 0.34, 0.92"   # P(P_i|V)
YOUNG_PROB = 0.1    # P(J);  P(V) = 1 - P(J)


def parse_probability_list(string : str):
    strings = string.replace("[","").replace("]","").split(',')
    probs = [float(s) for s in strings]
    [control_probability(p) for p in probs]
    return probs


@click.command(name="e1_1")
@click.option("--young-values", default=YOUNG_VALUES_DEFAULT)
@click.option("--old-values", default=OLD_VALUES_DEFAULT)
@click.option("--young-probability", default=YOUNG_PROB, type=float)
def main(young_values, old_values, young_probability):
    young_values = parse_probability_list(young_values)
    old_values = parse_probability_list(old_values)
    if len(old_values) != len(young_values):
        raise ValueError("Both values list must be equals in length")

    control_probability(young_probability)
    old_probability = 1 - young_probability

    values = np.array([young_values, old_values])   # write in array

    data = pd.DataFrame(values.T,
                        range(1, len(old_values) + 1),
                        ["P(Program_i|Young)",
                         "P(Program_i|Old)"])   # note: vals are transposed
    data.index.name = "Program"    # add index column name

    # Add table column with probability that someone (young or old) likes the
    # program i (= P(P_i))
    prob_p_i = pd.DataFrame(np.array(np.zeros((5, 1)))) # initialize with zeros

    for i in range(0, len(prob_p_i) - 1):
        prob_p_i.iloc[i + 1] = (data.iloc[i][0]) * young_probability + \
                               (data.iloc[i][1]) * old_probability
        # P(P_i) = P(P_i|J)*P(J) + P(P_i|V)*P(V)
        # Eric comment => data*[young_probability, old_probability] ??

    data.insert(2, "P(Programa_i)", prob_p_i)   # append to dataset

    # Add tables columns for P(joven|P_i) and P(viejo|P_i)
    prob_young_p_i = pd.DataFrame(np.array(np.zeros((5, 1))))  # initialize with 0
    prob_old_p_i = pd.DataFrame(np.array(np.zeros((5, 1))))    # initialize with 0

    for i in range(0, len(prob_young_p_i) - 1):
        prob_young_p_i.iloc[i + 1] = ((data.iloc[i][0]) * young_probability) / data.iloc[i][2]
        # P(J|P_i) = (P(P_i|J)*P(J))/P(P_i)
        prob_old_p_i.iloc[i + 1] = ((data.iloc[i][1]) * old_probability) / data.iloc[i][2]
        # P(V|P_i) = (P(P_i|V)*P(V))/P(P_i)
    # Eric comment => ((data.iloc[i][0]) * young_probability) => (Sub set * constant) * array (1/value saved)

    data.insert(3, "P(Joven|Programa_i)", prob_young_p_i)   # append to dataset
    data.insert(4, "P(Viejo|Programa_i)", prob_old_p_i)   # append to dataset

    # Compute prob for given example (hypothesis) TODO check if this hypothesis computation is correct
    example_prob_young = data.iloc[0][3]*data.iloc[0][2] + data.iloc[2][3]*data.iloc[2][2]
    # P(J) = P(J|P1)*P(P1) + P(J|P3)*P(P3)
    example_prob_old = data.iloc[0][4]*data.iloc[0][2] + data.iloc[2][4]*data.iloc[2][2]
    # P(V) = P(V|P1)*P(P1) + P(V|P3)*P(P3)

    # Final printout
    print("==================== Datos ====================")
    print("P(Joven) = ", young_probability, " | P(Viejo) = ", old_probability, "\n")
    print(data)

    print("\n==================== Resultados ====================")
    print("h_MAP(P(Joven|1,0,1,0)) = ", example_prob_young)
    print("h_MAP(P(Viejo|1,0,1,0)) = ", example_prob_old)