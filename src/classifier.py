from numpy import average
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.model_selection import cross_validate
from src.tokenizer import spacy_tokenizer
from src.csv_utils import commentparser, labelparser, write_stats
from src.keys import key_classifier, key_metric
import time


def classify():
    comments = commentparser()  # the features we want to analyze
    labels = labelparser()  # the labels, or answers, we want to test against

    tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)
    scores = {'ACCURACY': 'accuracy',
              'PRECISION': 'precision',
              'RECALL': 'recall',
              'F_MEASURE': 'f1'}

    stats = {}
    metric_names = []
    metric_values = []

    classifiers = [BernoulliNB, ComplementNB, MultinomialNB]
    stats[key_classifier] = [i.__name__ for i in classifiers]

    for i in classifiers:
        classifier = i()
        pipe = Pipeline([('vectorizer', tfidf_vector),
                         ('classifier', classifier)])

        result = cross_validate(pipe, comments, labels, cv=7, scoring=scores, return_train_score=True)

        print(i.__name__)

        if len(metric_names) == 0:
            metric_names = [x for x in result]

        for metric_name in metric_names:
            average_score = average(result[metric_name])
            metric_values.append(average_score)
            print('%s : %f' % (metric_name, average_score))
        print()

        if key_metric not in stats:
            stats[key_metric] = metric_names
        stats[i.__name__] = metric_values[:] #[round(x, 2) for x in metric_values]
        metric_values.clear()

    return stats


if __name__ == "__main__":
    start_time = time.time()
    write_stats(classify())
    print("--- %s seconds ---" % (time.time() - start_time))
