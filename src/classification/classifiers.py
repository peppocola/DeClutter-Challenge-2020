from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


class Classifier:
    def __init__(self, classifier, name=None):
        self.classifier = classifier
        if name is None:
            name = type(classifier).__name__
        self.name = name


def DummyClassifier_Initializer():
    return DummyClassifier(strategy='most_frequent')


def LinearSVC_Initializer():
    svm = LinearSVC()
    return CalibratedClassifierCV(svm)


def SDGClassifier_Initializer():
    return SGDClassifier(loss='log')


def SVC_rbf_Initializer():
    return SVC(kernel='rbf', probability=True)


def SVC_poly_Initializer(degree=2):
    return SVC(kernel='poly', degree=degree, probability=True)


def get_tf_idf_classifiers():
    return [Classifier(DummyClassifier_Initializer()),
            Classifier(BernoulliNB()),
            Classifier(ComplementNB()),
            Classifier(MultinomialNB()),
            Classifier(LinearSVC_Initializer(), LinearSVC.__name__),
            Classifier(SVC_rbf_Initializer(), SVC.__name__ + " (rbf)"),
            Classifier(SVC_poly_Initializer(), SVC.__name__ + " (poly degree=2)"),
            Classifier(MLPClassifier()),
            Classifier(RandomForestClassifier()),
            Classifier(AdaBoostClassifier()),
            Classifier(BaggingClassifier()),
            Classifier(ExtraTreesClassifier()),
            Classifier(GradientBoostingClassifier()),
            Classifier(LogisticRegression()),
            Classifier(DecisionTreeClassifier()),
            Classifier(SDGClassifier_Initializer())]


def get_feat_classifiers():
    return [Classifier(DummyClassifier_Initializer()),
            Classifier(BernoulliNB(alpha=0.0001, binarize=0.0, fit_prior=False)),
            Classifier(LinearSVC_Initializer(), LinearSVC.__name__),
            Classifier(SVC_rbf_Initializer(), SVC.__name__ + " (rbf)"),
            Classifier(SVC_poly_Initializer(), SVC.__name__ + " (poly degree=2)"),
            Classifier(MLPClassifier()),
            Classifier(RandomForestClassifier()),
            Classifier(AdaBoostClassifier()),
            Classifier(BaggingClassifier()),
            Classifier(ExtraTreesClassifier()),
            Classifier(GradientBoostingClassifier()),
            Classifier(LogisticRegression()),
            Classifier(DecisionTreeClassifier()),
            Classifier(SDGClassifier_Initializer())]


def get_tuned_classifiers():
    return [Classifier(DummyClassifier_Initializer()),
            Classifier(BernoulliNB(alpha=0.0001, binarize=0.0, fit_prior=False)),
            Classifier(LinearSVC_Initializer(), LinearSVC.__name__),
            Classifier(SVC_rbf_Initializer(), SVC.__name__ + " (rbf)"),
            Classifier(SVC_poly_Initializer(), SVC.__name__ + " (poly degree=2)"),
            Classifier(MLPClassifier()), # activation='logistic', alpha=0.5, hidden_layer_sizes=(50, 50, 50), learning_rate='adaptive', solver='adam'
            Classifier(RandomForestClassifier()),
            Classifier(AdaBoostClassifier(n_estimators=1200, learning_rate=0.1, algorithm='SAMME.R')),
            Classifier(BaggingClassifier(bootstrap=False, bootstrap_features=False, max_features=500, max_samples=0.5, n_estimators=200, warm_start=True)),
            Classifier(ExtraTreesClassifier()),
            Classifier(GradientBoostingClassifier()),
            Classifier(LogisticRegression()),
            Classifier(DecisionTreeClassifier()),
            Classifier(SDGClassifier_Initializer())]