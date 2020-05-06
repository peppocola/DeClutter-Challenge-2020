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

tf_idf_voting = [RandomForestClassifier, BernoulliNB, MLPClassifier, ExtraTreesClassifier, AdaBoostClassifier]

feat_classifiers = [BernoulliNB, LinearSVC, SVC, MLPClassifier, RandomForestClassifier,
                    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                    LogisticRegression, DecisionTreeClassifier, SGDClassifier]

feat_voting = [RandomForestClassifier, GradientBoostingClassifier, MLPClassifier, ExtraTreesClassifier,
               AdaBoostClassifier]

both_voting = [AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, SGDClassifier]


def tf_idf_classify(set='train', folder="tfidf-classifiers"):
    classifiers = tf_idf_classifiers
    voting = tf_idf_voting
    labels = get_labels(set=set)

    features = get_tfidf_features(set=set, normalized=False)

    return do_kfold(classifiers, voting, labels, features, folder)


def feat_classify(set='train', folder="features-classifiers", stemming=True, rem_kws=True, lines=None):
    classifiers = feat_classifiers
    voting = feat_voting

    features = get_features(set=set, stemming=stemming, rem_kws=rem_kws, scaled=True, lines=lines)
    labels = get_labels(set=set)

    return do_kfold(classifiers, voting, labels, features, folder)


def both_classify(set='train', folder="both-classifiers", stemming=True, rem_kws=True, lines=None):
    classifiers = feat_classifiers
    voting = both_voting

    features = get_both_features(set=set, stemming=stemming, rem_kws=rem_kws, lines=lines)
    labels = get_labels(set=set)

    return do_kfold(classifiers, voting, labels, features, folder)


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
    #data_split()

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
    train_n_test(tf_idf_classifiers, tf_idf_voting, dt_matrix_train, get_labels(set=train_set), dt_matrix_test, get_labels(set=test_set), folder=folder + "/tf_idf_classifier_tet/")

    # FEATURES
    lines_train = get_lines(serialized=False, serialize=True, set=train_set)

    feat_classify(set=train_set, folder=folder + "/feat_classifier/", lines=lines_train)

    features_train = get_features(set=train_set, scaled=True, lines=lines_train)
    lines_test = get_lines(serialized=False, serialize=True, set=test_set)
    features_test = get_features(set=test_set, scaled=True, lines=lines_test)

    train_n_test(feat_classifiers, feat_voting, features_train, get_labels(set=train_set), features_test,
                 get_labels(set=test_set), folder=folder + "/feat_classifier_tet/")

    # BOTH_CLASSIFIERS
    both_classify(set=train_set, folder=folder + "/both_classifier/", lines=lines_train)

    both_train = sparse.hstack((dt_matrix_train, features_train))
    both_test = sparse.hstack((dt_matrix_test, features_test))

    train_n_test(feat_classifiers, both_voting, both_train, get_labels(set=train_set), both_test,
                 get_labels(set=test_set), folder=folder + "/both_classifier_tet/")


def do_kfold(classifiers, voting, labels, features, folder):
    stats = {}
    list_results = []
    for i in classifiers:
        if i is DummyClassifier:
            classifier = i(strategy='most_frequent')
        else:
            classifier = i()
        predictions = cross_val_predict(classifier, features, labels,
                                        cv=KFold(n_splits=10, shuffle=True, random_state=seed))

        scores = cross_validate(estimator=classifier, scoring=scorers, X=features, y=labels,
                                cv=KFold(n_splits=10, shuffle=True, random_state=seed))
        if i in voting:
            list_results.append(predictions)
        cm = confusion_matrix(predictions, labels, [information, non_information])
        save_heatmap(cm, i.__name__, folder)

        print(i.__name__)
        # convert cross_validate report to a usable dict
        report = {}
        for name in scorers.keys():
            key = 'test_' + name
            report[name] = np.mean(scores[key])
        stats[i.__name__] = report
        print(report)
        stats[i.__name__] = report

    voting_results = []
    for i in range(len(labels)):
        vote = 0
        for j in range(len(voting)):
            vote += list_results[j][i]
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


def train_n_test(classifiers, voting, features_train, labels_train, features_test, labels_test, folder):
    list_results = []
    stats = {}

    for classifier in classifiers:
        c = classifier()
        c.fit(X=features_train, y=labels_train)
        result = c.predict(features_test)

        cm = confusion_matrix(result, labels_test, [information, non_information])
        save_heatmap(cm, classifier.__name__, folder)

        if classifier in voting:
            list_results.append(result)
        score = compute_metrics(labels_test, result)
        stats[classifier.__name__] = score
        print(classifier.__name__)
        print(score)

    voting_results = []
    for i in range(len(labels_test)):
        vote = 0
        for j in range(len(voting)):
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


if __name__ == "__main__":
    start_time = time.time()
    classify_split()
    # write_results(final_classify())
    #tf_idf_classify()
    #print("getting relevant lines")
    #lines = get_lines(serialized=True)
    #print("features\n")
    #feat_classify(lines=lines)
    #print("both\n")
    #both_classify(lines=lines)
    #print("--- %s seconds ---" % (time.time() - start_time))
