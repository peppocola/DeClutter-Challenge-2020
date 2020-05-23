from src.keys import non_information


def get_prediction_by_proba(proba, classes=None):
    if classes is None:
        classes = [0, 1]
    prediction = []
    for probability in proba:
        if probability[0] > 0.5:
            prediction.append(classes[0])
        else:
            prediction.append(classes[1])
    return prediction


class Probas:
    def __init__(self):
        self.dict_probas = {}
        self.dict_preds = {}

    def add_proba(self, proba, name):
        self.dict_probas[name] = proba
        self.dict_preds[name] = get_prediction_by_proba(proba)

    def get_proba(self, name):
        return self.dict_probas[name]

    def get_list_proba(self, name): #get the list of proba given by a classifier (name) for all the examples for the class 1 (non info yes)
        proba = []
        for probability in self.dict_probas[name]:
            proba.append(probability[non_information])
        return proba

    def get_proba_name_index(self, name, index): #get the probability given by a classifier (name) for an example (index) for the class 1 (non info yes)
        return self.dict_probas[name][index][non_information]

    def get_pred_name_index(self, name, index): #get the prediction given by a classifier (name) for an example (index)
        return self.dict_preds[name][index]

    def get_pred(self, name):
        return self.dict_preds[name]

    def get_names(self):
        return list(self.dict_probas.keys())

    def get_no_examples(self):
        return len(
            self.dict_probas[
                list(self.dict_probas.keys())
                [0]
            ]
        )
