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
    # Parse arguments
    young_values = np.array(parse_probability_list(young_values))
    old_values = np.array(parse_probability_list(old_values))
    if len(old_values) != len(young_values):
        raise ValueError("Both value lists must be equal in length")

    # Save probabilities
    control_probability(young_probability)
    old_probability = 1 - young_probability

    # Value matrix
    values = np.array([young_values, old_values]).T   # write in array and transpose

    data = pd.DataFrame(values,
                        range(1, len(old_values) + 1),
                        ["P(Program_i|Young)",
                         "P(Program_i|Old)"])
    data.index.name = "Program"    # add index column name

    # Add table column with probability that someone at all (young or old) likes a given program
    # P(P_i) = P(P_i|Y)*P(Y) + P(P_i|O)*P(O)
    # P(P_i) = [P(P_i|Y),P(P_i|O)].[P(Y),P(O)] -> matrix multiplication
    program_probability = np.dot(values, np.array([young_probability, old_probability]))

    data.insert(2, "P(Program_i)", program_probability)   # append to dataset

    # P(Y|P_i) = (P(P_i|Y) * P(Y))/P(P_i)
    p_young_as_p_i = (young_values * young_probability) / program_probability
    # P(O|P_i) = (P(P_i|O)*P(O))/P(P_i)
    p_old_as_p_i = (old_values * old_probability) / program_probability

    data.insert(3, "P(Young|Program_i)", p_young_as_p_i)   # append to dataset
    data.insert(4, "P(Old|Program_i)", p_old_as_p_i)   # append to dataset

    # Compute prob for given example (hypothesis)
    # P(Y | [1,0,1,0] ) = P(Program1=1|Young) * P(Program2=0|Young) * P(Program3=1|Young) * P(Program3=0|Young) * P(Young)
    p_y_likelihood = young_values[0] * (1-young_values[1]) * young_values[2] * (1-young_values[0]) * young_probability

    # P(O | [1,0,1,0] ) = P(Program1=1|Old) * P(Program2=0|Old) * P(Program3=1|Old) * P(Program3=0|Old) * P(Old)
    p_o_likelihood = old_values[0] * (1-old_values[1]) * old_values[2] * (1-old_values[3]) * old_probability

    # Final printout
    print("==================== Data ====================")
    print("P(Young) = ", young_probability, " | P(Old) = ", old_probability, "\n")
    print(data)

    print("\n================== Results ===================")
    print("P(Young|1,0,1,0) = ", p_y_likelihood)
    print("P(Old|1,0,1,0) = ", p_o_likelihood)

if __name__ == '__main__':
    main()
