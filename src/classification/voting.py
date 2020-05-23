import json
from collections import defaultdict

from sklearn.metrics import confusion_matrix

from src.classification.metrics import compute_metrics
from src.keys import non_information, information
from src.plot.plot_utils import save_heatmap
import numpy as np


def voting_selection(stats, criteria='f1_yes', n=5):
    classifiers = []
    f1_yes = []
    keys = list(stats.keys())
    for key in keys:
        f1_yes.append(stats[key][criteria])
    index = np.argpartition(f1_yes, -n)[-n:]

    for i in index:
        classifiers.append(keys[i])
    return classifiers


def compute_voting(voting, probas, labels, folder, voting_type='hard'):
    voting_results = []
    name = voting_type.lower()

    print("Voting=", voting)

    if voting_type == 'soft':
        get_vote = probas.get_proba_name_index
    elif voting_type == 'hard':
        get_vote = probas.get_pred_name_index
    else:
        raise ValueError

    for i in range(probas.get_no_examples()):
        vote = 0
        for j in voting:
            vote += get_vote(j, i)
        if vote > (len(voting) / 2):
            voting_results.append(non_information)
        else:
            voting_results.append(information)

    if labels is not None:
        print(name + '_voting')
        voting_report = compute_metrics(labels, voting_results)
        print(voting_report)
        cm = confusion_matrix(voting_results, labels, [information, non_information])
        save_heatmap(cm, name, folder)
    else:
        voting_report = {}
    return voting_report, voting_results


def dict_voting():
    x = open('../serialized_' + "voting" + '.json', 'r').read()
    listy = json.loads(x)
    dictionary = defaultdict(int)
    for row in listy:
        for classifier in row:
            dictionary[classifier] += 1
    x = open('../serialized_' + "voting_dict" + '.json', 'w')
    x.write(json.dumps(dictionary))
