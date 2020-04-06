from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.tree import DecisionTreeClassifier
from src.code_parser import tokenizer, get_positions_encoded, get_lines
from src.csv_utils import get_comments, get_labels, write_stats
from src.keys import non_information, information
from src.plot_utils import save_heatmap
from src.features_utils import jaccard, get_comment_length, get_links_tag, get_type_encoded

import numpy as np
from scipy import sparse

import time


def classify(classifiers=None, folder="tfidf-classifiers"):
    if classifiers is None:
        classifiers = [DummyClassifier, BernoulliNB, ComplementNB, MultinomialNB, LinearSVC, SVC, MLPClassifier,
                       RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                       GradientBoostingClassifier, LogisticRegression, DecisionTreeClassifier, SGDClassifier]

    voting = [RandomForestClassifier, BernoulliNB, MLPClassifier, ExtraTreesClassifier, LinearSVC]
    comments = get_comments()  # the features we want to analyze
    labels = get_labels()  # the labels, or answers, we want to testers against

    tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, lowercase=False)
    features = tfidf_vector.fit_transform(comments)
    stats = {}
    list_results = []
    for i in classifiers:
        if i is DummyClassifier:
            classifier = i(strategy='most_frequent')
        else:
            classifier = i()
        result = cross_val_predict(classifier, features, labels, cv=KFold(n_splits=10, shuffle=True))
        if i in voting:
            list_results.append(result)
        cm = confusion_matrix(result, labels, [information, non_information])
        save_heatmap(cm, i.__name__, folder)

        print(i.__name__)
        report = classification_report(labels, result, digits=3, target_names=['no', 'yes'], output_dict=True)
        report[matthews_corrcoef.__name__] = matthews_corrcoef(labels, result)
        print(report)
        stats[i.__name__] = report

    voting_results = []
    for i in range(len(comments)):
        vote = 0
        for j in range(len(voting)):
            vote += list_results[j][i]
        if vote > (len(voting) / 2):
            voting_results.append(non_information)
        else:
            voting_results.append(information)

    voting_report = classification_report(labels, voting_results, digits=3, target_names=['no', 'yes'],
                                          output_dict=True)
    voting_report[matthews_corrcoef.__name__] = matthews_corrcoef(labels, voting_results)
    cm = confusion_matrix(voting_results, labels, [information, non_information])
    save_heatmap(cm, "Voting", folder)
    stats["Voting"] = voting_report
    write_stats(stats, folder)
    return stats


def feat_classify(classifiers=None, folder="features-classifiers", stemming=True, rem_kws=True, jacc_score=None,
                  positions=None):
    if classifiers is None:
        classifiers = [BernoulliNB, ComplementNB, MultinomialNB, LinearSVC, SVC, MLPClassifier, RandomForestClassifier,
                       AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                       LogisticRegression, DecisionTreeClassifier, SGDClassifier]
    if jacc_score is None:
        jacc_score = jaccard(stemming, rem_kws)
    if positions is None:
        positions = get_positions_encoded()
    length = [x / 100 for x in get_comment_length()]
    types = get_type_encoded()
    features = []
    for i in range(len(length)):
        features.append([jacc_score[i], length[i], types[i], positions[i]])
    labels = get_labels()

    stats = {}
    for classifier in classifiers:
        result = cross_val_predict(classifier(), features, labels, cv=KFold(n_splits=10, shuffle=True))

        cm = confusion_matrix(result, labels, [information, non_information])
        save_heatmap(cm, classifier.__name__, folder)

        print(classifier.__name__)
        report = classification_report(labels, result, digits=3, target_names=['no', 'yes'], output_dict=True)
        report[matthews_corrcoef.__name__] = matthews_corrcoef(labels, result)
        print(report)
        stats[classifier.__name__] = report
    write_stats(stats, folder)
    return stats


def both_classify(classifiers=None, folder="both-classifiers", stemming=True, rem_kws=True, jacc_score=None,
                  positions=None):
    if classifiers is None:
        classifiers = [BernoulliNB, ComplementNB, MultinomialNB, LinearSVC, SVC, MLPClassifier, RandomForestClassifier,
                       AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                       LogisticRegression, DecisionTreeClassifier, SGDClassifier]
    if jacc_score is None:
        jacc_score = np.array(jaccard(stemming, rem_kws))
    if positions is None:
        positions = np.array(get_positions_encoded())
    length = np.array([x/100 for x in get_comment_length()]) # the x/100 makes some classifiers work worse
    types = np.array(get_type_encoded())
    comments = get_comments()
    labels = get_labels()

    tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, lowercase=False)
    dt_matrix = tfidf_vector.fit_transform(comments)
    features = sparse.hstack((dt_matrix, length.reshape((length.shape[0], 1))))
    features = sparse.hstack((features, jacc_score.reshape((length.shape[0], 1))))
    features = sparse.hstack((features, types.reshape((length.shape[0], 1))))
    features = sparse.hstack((features, positions.reshape((length.shape[0], 1))))
    stats = {}
    for classifier in classifiers:
        result = cross_val_predict(classifier(), features, labels, cv=KFold(n_splits=10, shuffle=True))

        cm = confusion_matrix(result, labels, [information, non_information])
        save_heatmap(cm, classifier.__name__, folder)

        print(classifier.__name__)
        report = classification_report(labels, result, digits=3, target_names=['no', 'yes'], output_dict=True)
        report[matthews_corrcoef.__name__] = matthews_corrcoef(labels, result)
        print(report)
        stats[classifier.__name__] = report
    write_stats(stats, folder)
    return stats


if __name__ == "__main__":
    start_time = time.time()
    print("tfidf -- BASELINE\n")
    classify()
    print("getting relevant lines")
    lines = get_lines(serialized=True)
    print("getting positions")
    positions = get_positions_encoded(lines=lines)
    print("jacc score calc")
    jacc = np.array(jaccard(lines=lines))
    print("features\n")
    feat_classify(jacc_score=jacc, positions=positions)
    print("both\n")
    both_classify(jacc_score=jacc, positions=positions)
    print("--- %s seconds ---" % (time.time() - start_time))
