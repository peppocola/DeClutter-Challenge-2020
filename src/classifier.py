from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, \
    accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_predict, KFold
from src.tokenizer import spacy_tokenizer
from src.csv_utils import commentparser, labelparser, write_stats
from src.keys import key_classifier, key_metric, non_information, information
from src.plot_utils import saveHeatmap

import time

# TODO : add precision-recall curve and ROC ---> FOR FUN
# TODO : create object stats
metrics = [accuracy_score, average_precision_score, recall_score, precision_score, f1_score]
metric_names = [x.__name__ for x in metrics]


def custom_metrics(cm):
    precision_no = str(round(cm[0][0] / (cm[0][0] + cm[0][1]), 2))
    recall_no = str(round(cm[0][0] / (cm[0][0] + cm[1][0]), 2))
    precision_yes = str(round(cm[1][1] / (cm[1][1] + cm[1][0]), 2))
    recall_yes = str(round(cm[1][1] / (cm[1][1] + cm[0][1]), 2))
    c_metrics = []
    c_metrics.extend([(precision_no, 'precision_no'), (recall_no, 'recall_no'),
                      (precision_yes, 'precision_yes'), (recall_yes, 'recall_yes')])
    return c_metrics


def custom_metrics_names():
    return ['precision_no',  'recall_no', 'precision_yes', 'recall_yes']


def classify(classifiers=None):
    if classifiers is None:
        classifiers = [BernoulliNB, ComplementNB, MultinomialNB, LinearSVC, SVC]
    comments = commentparser()  # the features we want to analyze
    labels = labelparser()  # the labels, or answers, we want to test against

    tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)
    for i in custom_metrics_names():
        metric_names.append(i)
    stats = {key_classifier: [i.__name__ for i in classifiers], key_metric: metric_names[:]}

    for i in classifiers:
        classifier = i()
        pipe = Pipeline([('vectorizer', tfidf_vector),
                         ('classifier', classifier)])

        result = cross_val_predict(pipe, comments, labels, cv=KFold(n_splits=10, shuffle=True))

        cm = confusion_matrix(result, labels, [information, non_information])
        saveHeatmap(cm, i.__name__)

        print(i.__name__)
        stats[i.__name__] = []
        for metric in metrics:
            stats[i.__name__].append(str(round(metric(result, labels), 2)))
        other_metrics = custom_metrics(cm)
        for metric in other_metrics:
            stats[i.__name__].append(metric[0])
    return stats


if __name__ == "__main__":
    start_time = time.time()
    write_stats(classify())
    print("--- %s seconds ---" % (time.time() - start_time))
