import numpy as np
import pandas as pd

print("==================== Data ====================\n")
dataset = pd.read_excel("Data/PreferenciasBritanicos.xlsx")
print(dataset)

# Calculate P(E) and P(I)
number_of_examples = len(dataset["Nacionalidad"])
prob_english, prob_irish = dataset["Nacionalidad"].value_counts()
prob_english /= number_of_examples
prob_irish /= number_of_examples
print("\nP(English) = ", prob_english, "\t|\tP(Irish) = ", prob_irish, "\n\n")

# Loop over all attributes associated --> P(a_i|v_j) = P(a_i AND v_j)/P(v_j)


# Calculate hypothesis for English and Scottish, respectively: Repeat-Multiply P(a_i|v_j)*P(v_j)


# Select higher scoring hypothesis for given example (1,0,1,1,0)
