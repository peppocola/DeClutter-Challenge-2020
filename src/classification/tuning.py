from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

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

"""
    #GRADIENT BOOSTING

    parameter_space = {
        'loss': ['deviance', 'exponential'],
        'learning_rate': [0.00001, 0.1, 0.5, 1],
        'n_estimators': [1, 10, 100, 1000],
        'subsample': [0.01, 0.1, 0.5, 1],
        'criterion': ['friedman_mse', 'mse', 'mae'],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [2, 10],
        'max_depth': [3, 10, 100],
        'max_features': ['auto', 'sqrt', 'log2'],
        'warm_start': [True, False]
    }
    clf = GridSearchCV(GradientBoostingClassifier(), parameter_space, 'f1_macro', n_jobs=-1, cv=KFold(n_splits=10), verbose=10)
    features = get_both_features()
    labels = get_labels()
    clf.fit(features, labels)
    print('Best parameters found:\n', clf.best_params_)
"""
"""
# GRADIENT BOOSTING 1

parameter_space = {'n_estimators':range(20,1000,20)}
clf = GridSearchCV(GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500, min_samples_leaf=50, max_depth=8, max_features='sqrt', subsample=0.8, random_state=10), parameter_space, 'f1_macro', n_jobs=-1, cv=KFold(n_splits=10),
                   verbose=10)
features = get_both_features()
labels = get_labels()
clf.fit(features, labels)
print('Best parameters found:\n', clf.best_params_)
print('Best score found:\n', clf.best_score_)
"""
"""
# GRADIENT BOOSTING 2

parameter_space = {'n_estimators': range(60, 661, 40), 'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
clf = GridSearchCV(GradientBoostingClassifier(learning_rate=0.1, min_samples_leaf=50, max_features='sqrt', subsample=0.8, random_state=10), parameter_space, 'f1_macro', n_jobs=-1, cv=KFold(n_splits=10),
                   verbose=10)
features = get_both_features()
labels = get_labels()
clf.fit(features, labels)
print('Best parameters found:\n', clf.best_params_)
print('Best score found:\n', clf.best_score_)
"""
"""
# GRADIENT BOOSTING 3
parameter_space = {'min_samples_split':range(600,1500,200), 'min_samples_leaf':range(20,100,10)}
clf = GridSearchCV(GradientBoostingClassifier(n_estimators=660, max_depth=5, learning_rate=0.1, max_features='sqrt', subsample=0.8, random_state=10), parameter_space, 'f1_macro', n_jobs=-1, cv=KFold(n_splits=10),
                   verbose=10)
features = get_both_features()
labels = get_labels()
clf.fit(features, labels)
print('Best parameters found:\n', clf.best_params_)
print('Best score found:\n', clf.best_score_)
"""
"""
# GRADIENT BOOSTING 4
parameter_space = {'max_features':range(2,200,2)}
clf = GridSearchCV(GradientBoostingClassifier(min_samples_split=600, min_samples_leaf=20, n_estimators=660, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=10), parameter_space, 'f1_macro', n_jobs=-1, cv=KFold(n_splits=10),
                   verbose=10)
features = get_both_features()
labels = get_labels()
clf.fit(features, labels)
print('Best parameters found:\n', clf.best_params_)
print('Best score found:\n', clf.best_score_)
"""
"""
# GRADIENT BOOSTING 5
parameter_space = {'subsample':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]}
clf = GridSearchCV(GradientBoostingClassifier(max_features=58, min_samples_split=600, min_samples_leaf=20, n_estimators=660, max_depth=5, learning_rate=0.1, random_state=10), parameter_space, 'f1_macro', n_jobs=-1, cv=KFold(n_splits=10),
                   verbose=10)
features = get_both_features()
labels = get_labels()
clf.fit(features, labels)
print('Best parameters found:\n', clf.best_params_)
print('Best score found:\n', clf.best_score_)"""
"""
# GRADIENT BOOSTING 6
parameter_space = {'learning_rate':[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]}
clf = GridSearchCV(GradientBoostingClassifier(max_features=58, min_samples_split=600, min_samples_leaf=20, n_estimators=660, max_depth=5, subsample=0.7, random_state=10), parameter_space, 'f1_macro', n_jobs=-1, cv=KFold(n_splits=10),
                   verbose=10)
features = get_both_features()
labels = get_labels()
clf.fit(features, labels)
print('Best parameters found:\n', clf.best_params_)
print('Best score found:\n', clf.best_score_)
"""
"""
# GRADIENT BOOSTING 7
parameter_space = {'warm_start':[True, False], 'max_features':[58, 'sqrt']}
clf = GridSearchCV(GradientBoostingClassifier(learning_rate=0.1, min_samples_split=600, min_samples_leaf=20, n_estimators=660, max_depth=5, subsample=0.7, random_state=10), parameter_space, 'f1_macro', n_jobs=-1, cv=KFold(n_splits=10),
                   verbose=10)
features = get_both_features()
labels = get_labels()
clf.fit(features, labels)
print('Best parameters found:\n', clf.best_params_)
print('Best score found:\n', clf.best_score_)
"""
"""
#SGDCLASSIFIER (REDO)
parameter_space = {'penalty':['l1','l2','elasticnet'],
                   'alpha':[0.000001, 0.00001, 0.0001, 0.001, 0.01],
                   'tol': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1],
                   'epsilon': [0.000001, 0.00001, 0.0001, 0.001, 0.01],
                   'max_iter':range(20,1000,100),
                   'learning_rate':['constant', 'invscaling', 'optimal', 'adaptive'],
                   'eta0':[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]
                   }
clf = GridSearchCV(SGDClassifier(loss='log', random_state=10, fit_intercept=True), param_grid=parameter_space, 'f1_macro', n_jobs=-1, verbose=10)
features = get_both_features()
labels = get_labels()
clf.fit(features, labels)
print('Best parameters found:\n', clf.best_params_)
print('Best score found:\n', clf.best_score_)
"""
"""
#LINEARSVC
parameter_space = {'penalty':['l1', 'l2'],
                   'loss':['hinge', 'squared_hinge'],
                   'dual':[True, False],
                   'tol':[0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1],
                   'fit_intercept':[True, False],
                   'max_iter':range(100, 10000, 100)
                   }
clf = GridSearchCV(LinearSVC(random_state=10), parameter_space, 'f1_macro', n_jobs=-1, verbose=2)
features = get_both_features()
labels = get_labels()
clf.fit(features, labels)
print('Best parameters found:\n', clf.best_params_)
print('Best score found:\n', clf.best_score_)
"""

if __name__ == '__main__':
    tune()
