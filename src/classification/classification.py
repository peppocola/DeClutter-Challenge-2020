#SKLEARN
import json

from sklearn.feature_extraction.text import TfidfVectorizer

#LIBS
from random import randrange
from scipy import sparse

#MYCODE
from src.classification.classifiers import get_feat_classifiers, get_tf_idf_classifiers
from src.classification.kfold import do_kfold
from src.classification.train_n_test import train_n_test
from src.comment_analysis.parsing_utils import tokenizer, get_lines
from src.csv.csv_utils import get_comments, get_labels
from src.classification.features_utils import get_both_features, get_features, get_tfidf_features

import time


def tf_idf_classify(classifiers=None, set='train', folder="tfidf-classifiers", voting=True):
    if classifiers is None:
        classifiers = get_tf_idf_classifiers()

    labels = get_labels(set=set)

    features = get_tfidf_features(set=set, normalized=False)

    return do_kfold(classifiers, labels, features, folder, voting)


def feat_classify(classifiers=None, set='train', folder="features-classifiers", stemming=True, rem_kws=True, lines=None,
                  voting=True):
    if classifiers is None:
        classifiers = get_feat_classifiers()

    features = get_features(set=set, stemming=stemming, rem_kws=rem_kws, scaled=True, lines=lines)
    labels = get_labels(set=set)

    return do_kfold(classifiers, labels, features, folder, voting)


def both_classify(classifiers=None, set='train', folder="both-classifiers", stemming=True, rem_kws=True, lines=None,
                  voting=True):
    if classifiers is None:
        classifiers = get_feat_classifiers()

    features = get_both_features(set=set, stemming=stemming, rem_kws=rem_kws, lines=lines)
    labels = get_labels(set=set)

    return do_kfold(classifiers, labels, features, folder, voting)


def classify_split(folder="split_classifier"):
    # data_split()

    train_set = 'split_train'
    test_set = 'split_test'

    selected_for_voting = []

    # TF-IDF
    comments_train = get_comments(set=train_set)
    tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, lowercase=False, sublinear_tf=True)
    dt_matrix_train = tfidf_vector.fit_transform(comments_train)
    # dt_matrix_train = normalize(dt_matrix_train, norm='l1', axis=0)

    comments_test = get_comments(set=test_set)
    dt_matrix_test = tfidf_vector.transform(comments_test)
    # dt_matrix_test = normalize(dt_matrix_test, norm='l1', axis=0)

    stats, voting = tf_idf_classify(set=train_set, folder=folder + "/tf_idf_classifier/")

    selected_for_voting.append(voting)

    stats, voting = train_n_test(get_tf_idf_classifiers(), dt_matrix_train, get_labels(set=train_set), dt_matrix_test,
                                 get_labels(set=test_set), folder=folder + "/tf_idf_classifier_tet/")

    selected_for_voting.append(voting)

    # FEATURES
    lines_train = get_lines(serialized=True, set=train_set)

    stats, voting = feat_classify(set=train_set, folder=folder + "/feat_classifier/", lines=lines_train)

    selected_for_voting.append(voting)

    features_train = get_features(set=train_set, scaled=True, lines=lines_train)
    lines_test = get_lines(serialized=True, set=test_set)
    features_test = get_features(set=test_set, scaled=True, lines=lines_test)

    stats, voting = train_n_test(get_feat_classifiers(), features_train, get_labels(set=train_set), features_test,
                                 get_labels(set=test_set), folder=folder + "/feat_classifier_tet/")

    selected_for_voting.append(voting)

    # BOTH_CLASSIFIERS
    stats, voting = both_classify(set=train_set, folder=folder + "/both_classifier/", lines=lines_train)

    selected_for_voting.append(voting)

    both_train = sparse.hstack((dt_matrix_train, features_train))
    both_test = sparse.hstack((dt_matrix_test, features_test))

    stats, voting = train_n_test(get_feat_classifiers(), both_train, get_labels(set=train_set), both_test,
                                 get_labels(set=test_set), folder=folder + "/both_classifier_tet/")

    selected_for_voting.append(voting)

    return selected_for_voting


if __name__ == "__main__":
    start_time = time.time()
    selected_for_voting = classify_split()
    stats, voting = tf_idf_classify()
    selected_for_voting.append(voting)
    print("getting relevant lines")
    lines = get_lines(serialized=True)
    print("features\n")
    stats, voting = feat_classify(lines=lines)
    selected_for_voting.append(voting)
    print("both\n")
    stats, voting = both_classify(lines=lines)

    selected_for_voting.append(voting)
    x = open('../serialized_' + "voting" + '.json', 'w')
    x.write(json.dumps(selected_for_voting))

    print("--- %s seconds ---" % (time.time() - start_time))
