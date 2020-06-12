from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_validate, KFold

from random import randrange
import numpy as np

from src.classification.metrics import scorers
from src.classification.probas import Probas
from src.classification.voting import voting_selection, compute_voting
from src.csv.csv_utils import write_stats
from src.keys import information, non_information
from src.plot.plot_utils import save_heatmap

#seed = randrange(100)
seed = 49
print("seed =", seed)


def do_kfold(classifiers, labels, features, folder, voting=True):
    probas = Probas()  # save proba for every classifier
    stats = {}

    for classifier in classifiers:
        clf = classifier.classifier
        result = cross_val_predict(clf, features, labels,
                                   cv=KFold(n_splits=10, shuffle=True, random_state=seed), method='predict_proba', n_jobs=-1)
        scores = cross_validate(estimator=clf, scoring=scorers, X=features, y=labels,
                                cv=KFold(n_splits=10, shuffle=True, random_state=seed), n_jobs=-1)

        probas.add_proba(result, classifier.name)

        cm = confusion_matrix(probas.get_pred(classifier.name), labels, [information, non_information])
        save_heatmap(cm, classifier.name, folder)

        print(classifier.name)
        # convert cross_validate report to a usable dict
        report = {}
        for name in scorers.keys():
            key = 'test_' + name
            report[name] = np.mean(scores[key])
        stats[classifier.name] = report
        print(report)
        stats[classifier.name] = report
    voting_set = []
    if voting:
        voting_set = voting_selection(stats)

        voting_reportW, voting_resultsW = compute_voting(voting_set, probas, labels, folder, 'soft')
        voting_reportN, voting_resultsN = compute_voting(voting_set, probas, labels, folder, 'hard')
        stats["VotingW"] = voting_reportW
        stats["VotingN"] = voting_reportN

    write_stats(stats, folder)
    return stats, voting_set

