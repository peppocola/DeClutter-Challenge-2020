from sklearn.ensemble import AdaBoostClassifier

from src.classification.features_utils import get_best_rfe_features, get_best_sfs_features, get_nt_feature_names, \
    get_mi_best_features
from src.keys import feature_selection_path

if __name__ == '__main__':
    f = open(feature_selection_path + 'rfe_ada.txt', 'w')
    f.write(str(get_best_rfe_features(AdaBoostClassifier, feature_type='nontextual')))
    f.close()

    f = open(feature_selection_path + 'sfs_ada_fwd.txt', 'w')
    f.write(str((get_best_sfs_features(forward=True, classifier=AdaBoostClassifier, feature_type='nontextual',
                                       k=(1,len(get_nt_feature_names()))))))
    f.close()

    f = open(feature_selection_path + 'sfs_ada_bck.txt', 'w')
    f.write(str((get_best_sfs_features(forward=True, classifier=AdaBoostClassifier, feature_type='nontextual',
                                       k=(1,len(get_nt_feature_names()))))))
    f.close()

    f = open(feature_selection_path + 'mutual_inf.txt', 'w')
    f.write(str(get_nt_feature_names()))
    f.write('\n')
    f.write(str(get_mi_best_features()))
    f.close()
