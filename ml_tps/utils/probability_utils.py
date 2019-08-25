
def control_probability(value):
    if value < 0 or 1 < value:
        raise ValueError("Probability out of range [0,1]")