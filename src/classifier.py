from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.tree import DecisionTreeClassifier

from src.code_parser import tokenizer
from src.csv_utils import get_comments, get_labels, write_stats
from src.keys import non_information, information
from src.plot_utils import save_heatmap
from src.feature_extractor import jaccard, get_comment_length, get_links_tag

import time


def classify(classifiers=None, folder="tfidf-classifiers"):
    if classifiers is None:
        classifiers = [BernoulliNB, ComplementNB, MultinomialNB, LinearSVC, SVC, MLPClassifier, RandomForestClassifier,
                       AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                       LogisticRegression, DecisionTreeClassifier, SGDClassifier]
    voting = [RandomForestClassifier, BernoulliNB, MLPClassifier, ExtraTreesClassifier, LinearSVC]
    comments = get_comments()  # the features we want to analyze
    labels = get_labels()  # the labels, or answers, we want to testers against

    tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, lowercase=False)
    stats = {}
    list_results = []
    for i in classifiers:
        classifier = i()
        pipe = Pipeline(
            [('vectorizer', tfidf_vector), ('classifier', classifier)])

        result = cross_val_predict(pipe, comments, labels, cv=KFold(n_splits=10, shuffle=True))
        if i in voting:
            list_results.append(result)
        cm = confusion_matrix(result, labels, [information, non_information])
        save_heatmap(cm, i.__name__, folder)

        print(i.__name__)
        report = classification_report(labels, result, digits=3, target_names=['no', 'yes'], output_dict=True)
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
    cm = confusion_matrix(voting_results, labels, [information, non_information])
    save_heatmap(cm, "Voting", folder)
    stats["Voting"] = voting_report
    write_stats(stats, folder)
    return stats


def feat_classify(classifiers=None, folder="features-classifiers", stemming=True, rem_kws=True):
    if classifiers is None:
        classifiers = [BernoulliNB, ComplementNB, MultinomialNB, LinearSVC, SVC, MLPClassifier, RandomForestClassifier,
                       AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                       LogisticRegression, DecisionTreeClassifier, SGDClassifier]
    jacc_score = jaccard(stemming, rem_kws)
    length = [x / 100 for x in get_comment_length()]
    links_tag = get_links_tag()
    features = []
    for i in range(len(length)):
        features.append([jacc_score[i], length[i], links_tag[i]])
    labels = get_labels()

    stats = {}
    for i in classifiers:
        classifier = i()
        pipe = Pipeline([('classifier', classifier)])

        result = cross_val_predict(pipe, features, labels, cv=KFold(n_splits=10, shuffle=True))

        cm = confusion_matrix(result, labels, [information, non_information])
        save_heatmap(cm, i.__name__, folder)

        print(i.__name__)
        report = classification_report(labels, result, digits=3, target_names=['no', 'yes'], output_dict=True)
        print(report)
        stats[i.__name__] = report
    write_stats(stats, folder)
    return stats


if __name__ == "__main__":
    start_time = time.time()
    #classify()
    feat_classify(stemming=False, rem_kws=False)
    print("--- %s seconds ---" % (time.time() - start_time))
