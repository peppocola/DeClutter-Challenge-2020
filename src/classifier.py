from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, \
    accuracy_score, classification_report
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


def classify(classifiers=None):
    if classifiers is None:
        classifiers = [BernoulliNB, ComplementNB, MultinomialNB, LinearSVC, SVC]
    comments = commentparser()  # the features we want to analyze
    labels = labelparser()  # the labels, or answers, we want to testers against

    tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)
    stats = {}
    for i in classifiers:
        classifier = i()
        pipe = Pipeline([('vectorizer', tfidf_vector),
                         ('classifier', classifier)])

        result = cross_val_predict(pipe, comments, labels, cv=KFold(n_splits=10, shuffle=True))

        cm = confusion_matrix(result, labels, [information, non_information])
        saveHeatmap(cm, i.__name__)

        print(i.__name__)
        report = classification_report(labels, result, digits=3, target_names=['no', 'yes'], output_dict=True)
        print(report)
        stats[i.__name__] = report
    print(stats)
    return stats


if __name__ == "__main__":
    start_time = time.time()
    write_stats(classify())
    print("--- %s seconds ---" % (time.time() - start_time))
