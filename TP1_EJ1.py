import numpy as np
import pandas as pd

vals_joven = [0.95, 0.05, 0.02, 0.20]   # P(P_i|J)
vals_viejo = [0.03, 0.82, 0.34, 0.92]   # P(P_i|V)
prob_joven = 0.1    # P(J)
prob_viejo = 0.9    # P(V)
vals = np.array([vals_joven, vals_viejo])   # write in array

data = pd.DataFrame(vals.T, [1, 2, 3, 4], ["P(Programa_i|Joven)", "P(Programa_i|Viejo)"])   # note: vals are transposed
data.index.name = "Programa"    # add index column name

# Add table column with probability that someone (young or old) likes program i (= P(P_i))
prob_p_i = pd.DataFrame(np.array(np.zeros((5, 1))))     # initialize with zeros

for i in range(0, len(prob_p_i) - 1):
    prob_p_i.iloc[i + 1] = (data.iloc[i][0]) * prob_joven + (data.iloc[i][1]) * prob_viejo   # P(P_i) = P(P_i|J)*P(J) + P(P_i|V)*P(V)

data.insert(2, "P(Programa_i)", prob_p_i)   # append to dataset

# Add tables columns for P(joven|P_i) and P(viejo|P_i)
prob_joven_p_i = pd.DataFrame(np.array(np.zeros((5, 1))))    # initialize with zeros
prob_viejo_p_i = pd.DataFrame(np.array(np.zeros((5, 1))))    # initialize with zeros

for i in range(0, len(prob_joven_p_i) - 1):
    prob_joven_p_i.iloc[i + 1] = ((data.iloc[i][0]) * prob_joven) / data.iloc[i][2]  # P(J|P_i) = (P(P_i|J)*P(J))/P(P_i)
    prob_viejo_p_i.iloc[i + 1] = ((data.iloc[i][1]) * prob_viejo) / data.iloc[i][2]  # P(V|P_i) = (P(P_i|V)*P(V))/P(P_i)

data.insert(3, "P(Joven|Programa_i)", prob_joven_p_i)   # append to dataset
data.insert(4, "P(Viejo|Programa_i)", prob_viejo_p_i)   # append to dataset

# Compute prob for given example (hypothesis) TODO check if this hypothesis computation is correct
example_prob_joven = data.iloc[0][3]*data.iloc[0][2] + data.iloc[2][3]*data.iloc[2][2]  # P(J) = P(J|P1)*P(P1) + P(J|P3)*P(P3)
example_prob_viejo = data.iloc[0][4]*data.iloc[0][2] + data.iloc[2][4]*data.iloc[2][2]  # P(V) = P(V|P1)*P(P1) + P(V|P3)*P(P3)

# Final printout
print("==================== Datos ====================")
print("P(Joven) = ", prob_joven, " | P(Viejo) = ", prob_viejo, "\n")
print(data)

print("\n==================== Resultados ====================")
print("h_MAP(P(Joven|1,0,1,0)) = ", example_prob_joven)
print("h_MAP(P(Viejo|1,0,1,0)) = ", example_prob_viejo)
