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
from src.code_parser import tokenizer, get_lines
from src.csv_utils import get_comments, get_labels, write_stats, data_split
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

tf_idf_classifiers = [DummyClassifier, BernoulliNB, ComplementNB, MultinomialNB, LinearSVC, SVC, MLPClassifier,
                      RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                      GradientBoostingClassifier, LogisticRegression, DecisionTreeClassifier, SGDClassifier]

feat_classifiers = [BernoulliNB, LinearSVC, SVC, MLPClassifier, RandomForestClassifier,
                    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                    LogisticRegression, DecisionTreeClassifier, SGDClassifier]


def tf_idf_classify(set='train', folder="tfidf-classifiers"):
    classifiers = tf_idf_classifiers

    labels = get_labels(set=set)

    features = get_tfidf_features(set=set, normalized=False)

    return do_kfold(classifiers, labels, features, folder)


def feat_classify(set='train', folder="features-classifiers", stemming=True, rem_kws=True, lines=None):
    classifiers = feat_classifiers

    features = get_features(set=set, stemming=stemming, rem_kws=rem_kws, scaled=True, lines=lines)
    labels = get_labels(set=set)

    return do_kfold(classifiers, labels, features, folder)


def both_classify(set='train', folder="both-classifiers", stemming=True, rem_kws=True, lines=None):
    classifiers = feat_classifiers

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
    train_n_test(tf_idf_classifiers, dt_matrix_train, get_labels(set=train_set), dt_matrix_test,
                 get_labels(set=test_set), folder=folder + "/tf_idf_classifier_tet/")

    # FEATURES
    lines_train = get_lines(serialized=True, set=train_set)

    feat_classify(set=train_set, folder=folder + "/feat_classifier/", lines=lines_train)

    features_train = get_features(set=train_set, scaled=True, lines=lines_train)
    lines_test = get_lines(serialized=True, set=test_set)
    features_test = get_features(set=test_set, scaled=True, lines=lines_test)

    train_n_test(feat_classifiers, features_train, get_labels(set=train_set), features_test,
                 get_labels(set=test_set), folder=folder + "/feat_classifier_tet/")

    # BOTH_CLASSIFIERS
    both_classify(set=train_set, folder=folder + "/both_classifier/", lines=lines_train)

    both_train = sparse.hstack((dt_matrix_train, features_train))
    both_test = sparse.hstack((dt_matrix_test, features_test))

    train_n_test(feat_classifiers, both_train, get_labels(set=train_set), both_test,
                 get_labels(set=test_set), folder=folder + "/both_classifier_tet/")


def do_kfold(classifiers, labels, features, folder):
    stats = {}
    list_results = {}
    for classifier in classifiers:
        if classifier is DummyClassifier:
            classifier_initialized = classifier(strategy='most_frequent')
        else:
            classifier_initialized = classifier()
        predictions = cross_val_predict(classifier_initialized, features, labels,
                                        cv=KFold(n_splits=10, shuffle=True, random_state=seed))

        scores = cross_validate(estimator=classifier_initialized, scoring=scorers, X=features, y=labels,
                                cv=KFold(n_splits=10, shuffle=True, random_state=seed))

        list_results[classifier.__name__] = predictions

        cm = confusion_matrix(predictions, labels, [information, non_information])
        save_heatmap(cm, classifier.__name__, folder)

        print(classifier.__name__)
        # convert cross_validate report to a usable dict
        report = {}
        for name in scorers.keys():
            key = 'test_' + name
            report[name] = np.mean(scores[key])
        stats[classifier.__name__] = report
        print(report)
        stats[classifier.__name__] = report

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

    voting_report = compute_metrics(labels, voting_results)

    cm = confusion_matrix(voting_results, labels, [information, non_information])
    save_heatmap(cm, "Voting", folder)
    stats["Voting"] = voting_report
    write_stats(stats, folder)
    return stats


def train_n_test(classifiers, features_train, labels_train, features_test, labels_test, folder):
    list_results = {}
    stats = {}

    for classifier in classifiers:
        classifier_initialized = classifier()
        classifier_initialized.fit(X=features_train, y=labels_train)
        result = classifier_initialized.predict(features_test)

        cm = confusion_matrix(result, labels_test, [information, non_information])
        save_heatmap(cm, classifier.__name__, folder)

        list_results[classifier.__name__] = result

        score = compute_metrics(labels_test, result)
        stats[classifier.__name__] = score
        print(classifier.__name__)
        print(score)

    voting_results = []
    voting = voting_selection(stats)
    print("Voting=", voting)

    for i in range(len(labels_test)):
        vote = 0
        for j in voting:
            vote += list_results[j][i]
        if vote > (len(voting) / 2):
            voting_results.append(non_information)
        else:
            voting_results.append(information)

    voting_report = compute_metrics(labels_test, voting_results)
    cm = confusion_matrix(voting_results, labels_test, [information, non_information])
    save_heatmap(cm, "Voting", folder)
    stats["Voting"] = voting_report
    write_stats(stats, folder)

    return voting_results


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


def map_classifiers(classifiers):
    new_classifiers = []
    for classifier_name in classifiers:
        for classifier in tf_idf_classifiers:
            if classifier == classifier_name:
                new_classifiers.append(classifier)
                break
    return new_classifiers


if __name__ == "__main__":
    start_time = time.time()
    classify_split()
    # write_results(final_classify())
    # tf_idf_classify()
    # print("getting relevant lines")
    # lines = get_lines(serialized=True)
    # print("features\n")
    # feat_classify(lines=lines)
    # print("both\n")
    # both_classify(lines=lines)
    # print("--- %s seconds ---" % (time.time() - start_time))
