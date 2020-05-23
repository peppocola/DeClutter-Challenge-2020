from sklearn.model_selection import GridSearchCV, KFold
from sklearn.naive_bayes import BernoulliNB

from src.classification.features_utils import get_both_features
from src.csv.csv_utils import get_labels


def tune():
    # MLP
    """parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive', 'invscaling']
    }
    clf = GridSearchCV(MLPClassifier(), parameter_space, 'f1_macro', n_jobs=-1, cv=10)
    features = get_both_features()
    labels = get_labels()
    clf.fit(features, labels)
    print('Best parameters found:\n', clf.best_params_)"""

    # ADAB -- U CAN TUNE ALSO BASE ESTIMATOR
    """parameter_space = {
        'n_estimators': [800, 1000, 1200, 1600, 2000, 3000, 4000],
        'learning_rate': [0.0001, 0.1],
        'algorithm': ['SAMME', 'SAMME.R'],
    }
    clf = GridSearchCV(AdaBoostClassifier(), parameter_space, 'f1_macro', n_jobs=-1, cv=KFold(n_splits=10), verbose=10)
    features = get_both_features()
    labels = get_labels()
    clf.fit(features, labels)
    print('Best parameters found:\n', clf.best_params_)"""

    # BAGGING -- U CAN TUNE ALSO BASE ESTIMATOR
    """parameter_space = {
        'n_estimators': [1, 5, 10, 20, 30, 40, 50, 100, 200, 500],
        'max_samples': [0.001, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10],
        'max_features': [1, 5, 10, 50, 100, 200, 500, 1000],
        'warm_start': [True, False],
        'bootstrap': [True, False],
        'bootstrap_features': [True, False]
    }
    clf = GridSearchCV(BaggingClassifier(), parameter_space, 'f1_macro', n_jobs=-1, cv=KFold(n_splits=10), verbose=10)
    features = get_both_features()
    labels = get_labels()
    clf.fit(features, labels)
    print('Best parameters found:\n', clf.best_params_)"""

    # BERNOULLI
    """parameter_space = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1],
        'fit_prior': [True, False],
        'binarize': [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1],
    }
    clf = GridSearchCV(BernoulliNB(), parameter_space, 'f1_macro', n_jobs=-1, cv=KFold(n_splits=10), verbose=10)
    features = get_both_features()
    labels = get_labels()
    clf.fit(features, labels)
    print('Best parameters found:\n', clf.best_params_)"""

    # GRADBOOST
    parameter_space = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1],
        'fit_prior': [True, False],
        'binarize': [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1],
    }
    clf = GridSearchCV(BernoulliNB(), parameter_space, 'f1_macro', n_jobs=-1, cv=KFold(n_splits=10), verbose=10)
    features = get_both_features()
    labels = get_labels()
    clf.fit(features, labels)
    print('Best parameters found:\n', clf.best_params_)


if __name__ == '__main__':
    tune()
