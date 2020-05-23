from sklearn.metrics import confusion_matrix

from src.classification.metrics import compute_metrics
from src.classification.probas import Probas
from src.classification.voting import voting_selection, compute_voting
from src.csv.csv_utils import write_stats
from src.keys import information, non_information
from src.plot.plot_utils import save_heatmap


def train_n_test(classifiers, features_train, labels_train, features_test, labels_test, folder, voting=True):
    probas = Probas()  # save proba for every classifier
    stats = {}

    for classifier in classifiers:
        clf = classifier.classifier
        clf.fit(X=features_train, y=labels_train)
        result = clf.predict_proba(features_test)

        probas.add_proba(result, classifier.name)

        cm = confusion_matrix(probas.get_pred(classifier.name), labels_test, [information, non_information])
        save_heatmap(cm, classifier.name, folder)

        score = compute_metrics(labels_test, probas.get_pred(classifier.name))
        stats[classifier.name] = score
        print(classifier.name)
        print(score)

    if voting:
        voting_set = voting_selection(stats)
        voting_reportW, voting_resultsW = compute_voting(voting_set, probas, labels_test, folder, 'soft')
        voting_reportN, voting_resultsN = compute_voting(voting_set, probas, labels_test, folder, 'hard')
        stats["VotingW"] = voting_reportW
        stats["VotingN"] = voting_reportN
    write_stats(stats, folder)

    return stats, voting