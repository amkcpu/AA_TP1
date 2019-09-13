import pandas

def subdateframe(dataset: pandas.DataFrame, attribute: str, value = None):
    if value is None:
        return [subdateframe(dataset,attribute=attribute,value=value) for value in dataset[attribute].unique()]
    return dataset[dataset[attribute] == value].drop(attribute,axis=1)