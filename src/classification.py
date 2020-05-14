from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from src.classifiers import get_feat_classifiers, get_tf_idf_classifiers
from src.code_parser import tokenizer, get_lines
from src.csv_utils import get_comments, get_labels, write_stats, write_results
from src.keys import non_information, information
from src.metrics import scorers, compute_metrics
from src.plot_utils import save_heatmap
from src.features_utils import get_both_features, get_features, get_tfidf_features
from random import randrange
import numpy as np
from scipy import sparse

import time

seed = randrange(100)
seed = 49
print("seed =", seed)


def tf_idf_classify(set='train', folder="tfidf-classifiers"):
    classifiers = get_tf_idf_classifiers()

    labels = get_labels(set=set)

    features = get_tfidf_features(set=set, normalized=False)

    return do_kfold(classifiers, labels, features, folder)


def feat_classify(set='train', folder="features-classifiers", stemming=True, rem_kws=True, lines=None):
    classifiers = get_feat_classifiers()

    features = get_features(set=set, stemming=stemming, rem_kws=rem_kws, scaled=True, lines=lines)
    labels = get_labels(set=set)

    return do_kfold(classifiers, labels, features, folder)


def both_classify(set='train', folder="both-classifiers", stemming=True, rem_kws=True, lines=None):
    classifiers = get_feat_classifiers()

    features = get_both_features(set=set, stemming=stemming, rem_kws=rem_kws, lines=lines)
    labels = get_labels(set=set)

    return do_kfold(classifiers, labels, features, folder)


def final_classify(stemming=True, rem_kws=True):
    features_train = get_features(set='train', stemming=stemming, rem_kws=rem_kws)

    comments_train = get_comments(set='train')
    tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, lowercase=False, sublinear_tf=True)
    dt_matrix_train = tfidf_vector.fit_transform(comments_train)
    dt_matrix_train = normalize(dt_matrix_train, norm='l1', axis=0)

    features_train = sparse.hstack((dt_matrix_train, features_train))

    comments_test = get_comments(set='split_test')
    features_test = get_features(set='split_test', stemming=stemming, rem_kws=rem_kws)
    dt_matrix_test = tfidf_vector.transform(comments_test)
    dt_matrix_test = normalize(dt_matrix_test, norm='l1', axis=0)

    features_test = sparse.hstack((dt_matrix_test, features_test))

    labels = get_labels(set='train')

    classifiers = [AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier,
                   SGDClassifier]
    results = []
    for classifier in classifiers:
        c = classifier()
        c.fit(X=features_train, y=labels)
        result = c.predict(features_test)
        results.append(result)
        print(result)

    voting_results = []
    for i in range(len(results[0])):
        vote = 0
        for j in range(len(results)):
            vote += results[j][i]
        if vote > (len(results) / 2):
            voting_results.append(non_information)
        else:
            voting_results.append(information)
    return voting_results


def classify_split(folder="split_classifier"):
    # data_split()

    train_set = 'split_train'
    test_set = 'split_test'

    # TF-IDF
    comments_train = get_comments(set=train_set)
    tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, lowercase=False, sublinear_tf=True)
    dt_matrix_train = tfidf_vector.fit_transform(comments_train)
    # dt_matrix_train = normalize(dt_matrix_train, norm='l1', axis=0)

    comments_test = get_comments(set=test_set)
    dt_matrix_test = tfidf_vector.transform(comments_test)
    # dt_matrix_test = normalize(dt_matrix_test, norm='l1', axis=0)

    tf_idf_classify(set=train_set, folder=folder + "/tf_idf_classifier/")
    train_n_test(get_tf_idf_classifiers(), dt_matrix_train, get_labels(set=train_set), dt_matrix_test,
                 get_labels(set=test_set), folder=folder + "/tf_idf_classifier_tet/")

    # FEATURES
    lines_train = get_lines(serialized=True, set=train_set)

    feat_classify(set=train_set, folder=folder + "/feat_classifier/", lines=lines_train)

    features_train = get_features(set=train_set, scaled=True, lines=lines_train)
    lines_test = get_lines(serialized=True, set=test_set)
    features_test = get_features(set=test_set, scaled=True, lines=lines_test)

    train_n_test(get_feat_classifiers(), features_train, get_labels(set=train_set), features_test,
                 get_labels(set=test_set), folder=folder + "/feat_classifier_tet/")

    # BOTH_CLASSIFIERS
    both_classify(set=train_set, folder=folder + "/both_classifier/", lines=lines_train)

    both_train = sparse.hstack((dt_matrix_train, features_train))
    both_test = sparse.hstack((dt_matrix_test, features_test))

    train_n_test(get_feat_classifiers(), both_train, get_labels(set=train_set), both_test,
                 get_labels(set=test_set), folder=folder + "/both_classifier_tet/")


def do_kfold(classifiers, labels, features, folder):
    probas = {} #save proba for every classifier
    preds = {} #save predictions for every classifier
    stats = {}

    for classifier in classifiers:
        clf = classifier.classifier
        result = cross_val_predict(clf, features, labels,
                                        cv=KFold(n_splits=10, shuffle=True, random_state=seed), method='predict_proba')
        predictions = get_predictions_by_proba(result)
        scores = cross_validate(estimator=clf, scoring=scorers, X=features, y=labels,
                                cv=KFold(n_splits=10, shuffle=True, random_state=seed))

        probas[classifier.name] = get_list_proba(result)
        preds[classifier.name] = predictions

        cm = confusion_matrix(predictions, labels, [information, non_information])
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

    voting_reportW, voting_resultsW = compute_voting(stats, probas, labels, folder, 'VotingW')
    voting_reportN, voting_resultsN = compute_voting(stats, preds, labels, folder, 'VotingN')
    stats["VotingW"] = voting_reportW
    stats["VotingN"] = voting_reportN
    write_stats(stats, folder)
    return stats


def train_n_test(classifiers, features_train, labels_train, features_test, labels_test, folder):
    probas = {} #save proba for every classifier
    preds = {} #save predictions for every classifier
    stats = {}

    for classifier in classifiers:
        clf = classifier.classifier
        clf.fit(X=features_train, y=labels_train)
        result = clf.predict_proba(features_test)
        predictions = get_predictions_by_proba(result, clf.classes_)
        cm = confusion_matrix(predictions, labels_test, [information, non_information])
        save_heatmap(cm, classifier.name, folder)

        probas[classifier.name] = get_list_proba(result)
        preds[classifier.name] = predictions

        score = compute_metrics(labels_test, predictions)
        stats[classifier.name] = score
        print(classifier.name)
        print(score)

    voting_reportW, voting_resultsW = compute_voting(stats, probas, labels_test, folder, 'VotingW')
    voting_reportN, voting_resultsN = compute_voting(stats, probas, labels_test, folder, 'VotingN')
    stats["VotingW"] = voting_reportW
    stats["VotingN"] = voting_reportN
    write_stats(stats, folder)

    return stats


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


def get_predictions_by_proba(probabilities, classes=None):
    if classes is None:
        classes = [0, 1]
    prediction = []
    for probability in probabilities:
        if probability[0] > 0.5:
            prediction.append(classes[0])
        else:
            prediction.append(classes[1])
    return prediction


def get_list_proba(probabilities):
    proba = []
    for probability in probabilities:
        proba.append(probability[1])
    return proba


def compute_voting(stats, list_results, labels, folder, name='Voting'):
    voting_results = []
    voting = voting_selection(stats)
    print("Voting=", voting)
    for classifier in range(len(labels)):
        vote = 0
        for j in voting:
            vote += list_results[j][classifier]
        if vote > (len(voting) / 2):
            voting_results.append(non_information)
        else:
            voting_results.append(information)

    print(name)
    voting_report = compute_metrics(labels, voting_results)
    print(voting_report)
    cm = confusion_matrix(voting_results, labels, [information, non_information])
    save_heatmap(cm, name, folder)

    return voting_report, voting_results


if __name__ == "__main__":
    start_time = time.time()
    classify_split()
    """write_results(final_classify())"""
    tf_idf_classify()
    print("getting relevant lines")
    lines = get_lines(serialized=True)
    print("features\n")
    feat_classify(lines=lines)
    print("both\n")
    both_classify(lines=lines)
    print("--- %s seconds ---" % (time.time() - start_time))
