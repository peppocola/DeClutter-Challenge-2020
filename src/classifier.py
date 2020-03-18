from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, KFold
from src.code_parser import code_split, tokenizer
from src.csv_utils import get_comments, get_labels, write_stats
from src.keys import non_information, information
from src.plot_utils import saveHeatmap

import time


def classify(classifiers=None):
    if classifiers is None:
        classifiers = [BernoulliNB, ComplementNB, MultinomialNB, LinearSVC, SVC, MLPClassifier, RandomForestClassifier,
                       AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier]
    voting = [RandomForestClassifier, BernoulliNB, MLPClassifier, ExtraTreesClassifier, LinearSVC]
    comments = get_comments()  # the features we want to analyze
    labels = get_labels()  # the labels, or answers, we want to testers against

    tfidf_vector = TfidfVectorizer(tokenizer=tokenizer)
    stats = {}
    list_results = []
    for i in classifiers:
        classifier = i()
        pipe = Pipeline([('vectorizer', tfidf_vector), ('classifier', classifier)])

        result = cross_val_predict(pipe, comments, labels, cv=KFold(n_splits=10, shuffle=True))
        if i in voting:
            list_results.append(result)
        cm = confusion_matrix(result, labels, [information, non_information])
        saveHeatmap(cm, i.__name__)

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
    saveHeatmap(cm, "Voting")
    stats["Voting"] = voting_report
    return stats


if __name__ == "__main__":
    start_time = time.time()
    write_stats(classify())
    print("--- %s seconds ---" % (time.time() - start_time))
