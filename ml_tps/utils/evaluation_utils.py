# F1-score
def f1_score(precision, recall):
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
