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
            Classifier(BernoulliNB()),
            Classifier(LinearSVC_Initializer(), LinearSVC.__name__),
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