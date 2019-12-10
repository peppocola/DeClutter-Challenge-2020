from numpy import average
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.model_selection import cross_validate
from tokenizer import spacy_tokenizer
from csv_utils import commentparser, labelparser


def classify():
    comments = commentparser()  # the features we want to analyze
    labels = labelparser()  # the labels, or answers, we want to test against

    tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)
    scores = {'acc': 'accuracy',
              'prec_macro': 'precision_macro',
              'rec_micro': 'recall_macro',
              'f1': 'f1'}

    for i in [BernoulliNB, ComplementNB, MultinomialNB]:
        classifier = i()
        pipe = Pipeline([('vectorizer', tfidf_vector),
                         ('classifier', classifier)])

        result = cross_validate(pipe, comments, labels, cv=7, scoring=scores, return_train_score=True)

        print(i.__name__)
        for metric_name in result.keys():
            average_score = average(result[metric_name])
            print('%s : %f' % (metric_name, average_score))
        print()


if __name__ == "__main__":
    classify()
