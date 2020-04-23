from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.preprocessing import scale, normalize
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.tree import DecisionTreeClassifier
from src.code_parser import tokenizer, get_positions_encoded, get_lines
from src.csv_utils import get_comments, get_labels, write_stats, write_results
from src.keys import non_information, information
from src.plot_utils import save_heatmap
from src.features_utils import jaccard, get_comment_length, get_links_tag, get_type_encoded, get_no_sep
from random import randrange
import numpy as np
from scipy import sparse

import time

seed = randrange(100)
print("seed =", seed)


def classify(classifiers=None, folder="tfidf-classifiers"):
    if classifiers is None:
        classifiers = [DummyClassifier, BernoulliNB, ComplementNB, MultinomialNB, LinearSVC, SVC, MLPClassifier,
                       RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                       GradientBoostingClassifier, LogisticRegression, DecisionTreeClassifier, SGDClassifier]

    voting = [RandomForestClassifier, BernoulliNB, MLPClassifier, ExtraTreesClassifier, AdaBoostClassifier]
    comments = get_comments()  # the features we want to analyze
    labels = get_labels()  # the labels, or answers, we want to testers against

    tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, lowercase=False, sublinear_tf=True)
    features = tfidf_vector.fit_transform(comments)
    stats = {}
    list_results = []
    for i in classifiers:
        if i is DummyClassifier:
            classifier = i(strategy='most_frequent')
        else:
            classifier = i()
        result = cross_val_predict(classifier, features, labels, cv=KFold(n_splits=10, shuffle=True, random_state=seed))
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
        """classifiers = [BernoulliNB, ComplementNB, MultinomialNB, LinearSVC, SVC, MLPClassifier, RandomForestClassifier,
                       AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                       LogisticRegression, DecisionTreeClassifier, SGDClassifier]""" #removed complement and multinomial cuz scaled
        classifiers = [BernoulliNB, LinearSVC, SVC, MLPClassifier, RandomForestClassifier,
                       AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                       LogisticRegression, DecisionTreeClassifier, SGDClassifier]
    voting = [RandomForestClassifier, GradientBoostingClassifier, MLPClassifier, ExtraTreesClassifier,
              AdaBoostClassifier]

    if jacc_score is None:
        jacc_score = jaccard(stemming, rem_kws)
    if positions is None:
        positions = get_positions_encoded()
    rough_length = [x / 100 for x in get_comment_length(rough=True)]
    length = [x for x in get_comment_length(rough=False)]
    types = get_type_encoded()
    link_tag = get_links_tag()
    no_sep = np.array(get_no_sep())

    features = []
    for i in range(len(rough_length)):
        features.append([jacc_score[i], rough_length[i], length[i], types[i], positions[i], link_tag[i], no_sep[i]])
    features = scale(features)

    labels = get_labels()
    stats = {}
    list_results = []
    for classifier in classifiers:
        result = cross_val_predict(classifier(), features, labels, cv=KFold(n_splits=10, shuffle=True, random_state=seed))
        if classifier in voting:
            list_results.append(result)
        cm = confusion_matrix(result, labels, [information, non_information])
        save_heatmap(cm, classifier.__name__, folder)

        print(classifier.__name__)
        report = classification_report(labels, result, digits=3, target_names=['no', 'yes'], output_dict=True)
        report[matthews_corrcoef.__name__] = matthews_corrcoef(labels, result)
        print(report)
        stats[classifier.__name__] = report
    voting_results = []
    for i in range(len(labels)):
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


def both_classify(classifiers=None, folder="both-classifiers", stemming=True, rem_kws=True, jacc_score=None,
                  positions=None):
    if classifiers is None:
        """classifiers = [BernoulliNB, ComplementNB, MultinomialNB, LinearSVC, SVC, MLPClassifier, RandomForestClassifier,
                       AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                       LogisticRegression, DecisionTreeClassifier, SGDClassifier]""" #removed complement and multinomial cuz scaled
        classifiers = [BernoulliNB, LinearSVC, SVC, MLPClassifier, RandomForestClassifier,
                       AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                       LogisticRegression, DecisionTreeClassifier, SGDClassifier]
    if jacc_score is None:
        jacc_score = np.array(jaccard(stemming, rem_kws))
    if positions is None:
        positions = np.array(get_positions_encoded())
    rough_length = np.array([x for x in get_comment_length(rough=True)])  # the x/100 makes some classifiers work worse
    length = np.array([x for x in get_comment_length(rough=False)])
    types = np.array(get_type_encoded())
    link_tag = np.array(get_links_tag())
    no_sep = np.array(get_no_sep(set='train'))
    comments = get_comments()
    labels = get_labels()

    tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, lowercase=False, sublinear_tf=True)
    dt_matrix = tfidf_vector.fit_transform(comments)
    #dt_matrix = dt_matrix / np.linalg.norm(dt_matrix)

    features = rough_length.reshape((rough_length.shape[0], 1))
    features = np.hstack((features, length.reshape((length.shape[0], 1))))
    features = np.hstack((features, jacc_score.reshape((jacc_score.shape[0], 1))))
    features = np.hstack((features, types.reshape((types.shape[0], 1))))
    features = np.hstack((features, positions.reshape((positions.shape[0], 1))))
    features = np.hstack((features, link_tag.reshape((link_tag.shape[0], 1))))
    features = np.hstack((features, no_sep.reshape((no_sep.shape[0], 1))))
    features = scale(features)
    features = sparse.hstack((dt_matrix, features))

    """features = sparse.hstack((dt_matrix, rough_length.reshape((rough_length.shape[0], 1))))
    features = sparse.hstack((features, length.reshape((length.shape[0], 1))))
    features = sparse.hstack((features, jacc_score.reshape((jacc_score.shape[0], 1))))
    features = sparse.hstack((features, types.reshape((types.shape[0], 1))))
    features = sparse.hstack((features, positions.reshape((positions.shape[0], 1))))
    features = sparse.hstack((features, link_tag.reshape((link_tag.shape[0], 1))))
    features = sparse.hstack((features, no_sep.reshape((no_sep.shape[0], 1))))"""
    stats = {}
    for classifier in classifiers:
        result = cross_val_predict(classifier(), features, labels, cv=KFold(n_splits=10, shuffle=True, random_state=seed))

        cm = confusion_matrix(result, labels, [information, non_information])
        save_heatmap(cm, classifier.__name__, folder)

        print(classifier.__name__)
        report = classification_report(labels, result, digits=3, target_names=['no', 'yes'], output_dict=True)
        report[matthews_corrcoef.__name__] = matthews_corrcoef(labels, result)
        print(report)
        stats[classifier.__name__] = report
    write_stats(stats, folder)
    return stats


def final_classify(stemming=True, rem_kws=True):
    lines = get_lines()
    jacc_score = np.array(jaccard(stemming, rem_kws, lines=lines))
    positions = np.array(get_positions_encoded(lines=lines))
    rough_length = np.array([x / 100 for x in get_comment_length(rough=True)])  # the x/100 makes some classifiers work worse
    length = np.array([x for x in get_comment_length(rough=False)])
    types = np.array(get_type_encoded())
    link_tag = np.array(get_links_tag())
    no_sep = np.array(get_no_sep(set='train'))
    comments = get_comments()
    labels = get_labels()

    tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, lowercase=False, sublinear_tf=True)
    dt_matrix = tfidf_vector.fit_transform(comments)
    #dt_matrix = dt_matrix / np.linalg.norm(dt_matrix)

    features = rough_length.reshape((rough_length.shape[0], 1))
    features = np.hstack((features, length.reshape((length.shape[0], 1))))
    features = np.hstack((features, jacc_score.reshape((jacc_score.shape[0], 1))))
    features = np.hstack((features, types.reshape((types.shape[0], 1))))
    features = np.hstack((features, positions.reshape((positions.shape[0], 1))))
    features = np.hstack((features, link_tag.reshape((link_tag.shape[0], 1))))
    features = np.hstack((features, no_sep.reshape((no_sep.shape[0], 1))))

    features = scale(features)
    features = sparse.hstack((dt_matrix, features))
    """
    features = sparse.hstack((dt_matrix, features))
    features = sparse.hstack((dt_matrix, rough_length.reshape((rough_length.shape[0], 1))))
    features = sparse.hstack((features, length.reshape((length.shape[0], 1))))
    features = sparse.hstack((features, jacc_score.reshape((jacc_score.shape[0], 1))))
    features = sparse.hstack((features, types.reshape((types.shape[0], 1))))
    features = sparse.hstack((features, positions.reshape((positions.shape[0], 1))))
    #features = sparse.hstack((features, no_sep.reshape((no_sep.shape[0], 1))))
    """
    print('test')

    comments_test = get_comments(set='test')
    lines_test = get_lines(serialized=True, serialize=False, set='test')
    rough_length_test = np.array([x for x in get_comment_length(set='test', rough=True)])
    length_test = np.array([x for x in get_comment_length(set='test', rough=False)])
    jacc_score_test = np.array(jaccard(stemming, rem_kws, lines=lines_test, set='test'))
    positions_test = np.array(get_positions_encoded(lines=lines_test, set='test'))
    types_test = np.array(get_type_encoded(set='test'))
    link_tag_test = np.array(get_links_tag(set='test'))
    no_sep_test = np.array(get_no_sep(set='test'))

    dt_matrix_test = tfidf_vector.transform(comments_test)

    #dt_matrix_test = dt_matrix_test / np.linalg.norm(dt_matrix_test)

    features_test = rough_length_test.reshape((rough_length_test.shape[0], 1))
    features_test = np.hstack((features_test, length_test.reshape((length_test.shape[0], 1))))
    features_test = np.hstack((features_test, jacc_score_test.reshape((jacc_score_test.shape[0], 1))))
    features_test = np.hstack((features_test, types_test.reshape((types_test.shape[0], 1))))
    features_test = np.hstack((features_test, positions_test.reshape((positions_test.shape[0], 1))))
    features_test = np.hstack((features_test, no_sep_test.reshape((no_sep_test.shape[0], 1))))
    features_test = np.hstack((features_test, link_tag_test.reshape((link_tag_test.shape[0], 1))))

    features_test = scale(features_test)

    features_test = sparse.hstack((dt_matrix_test, features_test))

    """
    features_test = sparse.hstack((dt_matrix_test, rough_length_test.reshape((rough_length_test.shape[0], 1))))
    features_test = sparse.hstack((features_test, length_test.reshape((length_test.shape[0], 1))))
    features_test = sparse.hstack((features_test, jacc_score_test.reshape((jacc_score_test.shape[0], 1))))
    features_test = sparse.hstack((features_test, types_test.reshape((types_test.shape[0], 1))))
    features_test = sparse.hstack((features_test, positions_test.reshape((positions_test.shape[0], 1))))
    features_test = sparse.hstack((features_test, no_sep_test.reshape((no_sep_test.shape[0], 1))))
    features_test = sparse.hstack((features_test, link_tag_test.reshape((link_tag_test.shape[0], 1))))
    """
    classifiers = [AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier,
                   ExtraTreesClassifier]
    results = []
    for classifier in classifiers:
        c = classifier()
        c.fit(X=features, y=labels)
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
    for i in zip(voting_results, comments_test):
        print(i)
    return voting_results


if __name__ == "__main__":
    start_time = time.time()
    write_results(final_classify())
    print("tfidf -- BASELINE\n")
    #classify()
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
