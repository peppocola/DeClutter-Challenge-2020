from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize

from scipy import sparse

from src.classification.classifiers import Classifier, LinearSVC_Initializer, SDGClassifier_Initializer
from src.classification.features_utils import get_features
from src.classification.probas import Probas
from src.classification.voting import compute_voting
from src.comment_analysis.parsing_utils import tokenizer, get_lines
from src.csv.csv_utils import get_comments, get_labels

tfidf_c = [Classifier(MLPClassifier()), Classifier(BaggingClassifier()), Classifier(BernoulliNB()), Classifier(ExtraTreesClassifier()),Classifier(RandomForestClassifier())]
feat_c = [Classifier(LogisticRegression()), Classifier(MLPClassifier()), Classifier(RandomForestClassifier()), Classifier(GradientBoostingClassifier()), Classifier(AdaBoostClassifier())]
both_c = [Classifier(SDGClassifier_Initializer()), Classifier(LinearSVC_Initializer()), Classifier(GradientBoostingClassifier()), Classifier(AdaBoostClassifier()), Classifier(BaggingClassifier())]


def kaggle_classify(stemming=True, rem_kws=True):
    train_set = 'def_train'
    test_set = 'def_test'

    lines_train = get_lines(serialized=True, set=train_set)
    features_train = get_features(set=train_set, stemming=stemming, rem_kws=rem_kws, lines=lines_train)

    comments_train = get_comments(set=train_set)

    #tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, lowercase=False, sublinear_tf=True)
    #dt_matrix_train = tfidf_vector.fit_transform(comments_train)

    count_vector = CountVectorizer(tokenizer=tokenizer, lowercase=False)
    dt_matrix_train = count_vector.fit_transform(comments_train)

    dt_matrix_train = normalize(dt_matrix_train, norm='l1', axis=0)

    features_train = sparse.hstack((dt_matrix_train, features_train))

    comments_test = get_comments(set=test_set)
    lines_test = get_lines(serialized=True, set=test_set)
    features_test = get_features(set=test_set, stemming=stemming, rem_kws=rem_kws, lines=lines_test)

    #dt_matrix_test = tfidf_vector.transform(comments_test)

    dt_matrix_test = count_vector.transform(comments_test)

    dt_matrix_test = normalize(dt_matrix_test, norm='l1', axis=0)

    features_test = sparse.hstack((dt_matrix_test, features_test))

    labels = get_labels(set='def_train')

    """classifiers = [Classifier(BaggingClassifier(bootstrap=False, bootstrap_features=False, max_features=500, max_samples=0.5, n_estimators=200, warm_start=True)), Classifier(LinearSVC_Initializer()), Classifier(SDGClassifier_Initializer()), Classifier(GradientBoostingClassifier()),
                   Classifier(AdaBoostClassifier(n_estimators=1200, learning_rate=0.1, algorithm='SAMME.R'))]"""
    classifiers = both_c
    results = Probas()
    for classifier in classifiers:
        classifier.classifier.fit(X=features_train, y=labels)
        result = classifier.classifier.predict_proba(features_test)
        results.add_proba(result, classifier.name)
        print(result)
        print(len(result))

    _, voting_results = compute_voting(voting=results.get_names(), probas=results, labels=None, folder=None, voting_type='soft')
    return voting_results, test_set
