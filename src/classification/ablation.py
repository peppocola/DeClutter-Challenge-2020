from scipy import sparse
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale, normalize
import numpy as np

from src.classification.classifiers import Classifier
from src.classification.features_utils import jaccard, get_comment_length, get_type_encoded, get_links_tag
from src.classification.kfold import do_kfold
from src.comment_analysis.parsing_utils import get_lines, get_positions_encoded, tokenizer
from src.csv.csv_utils import get_comments, get_labels


def ablation(set='def_train', stemming=True, rem_kws=True, scaled=True, normalized=True, lines=None):
    if lines is None:
        lines = get_lines(set=set)

    #NON-TEXTUAL
    jacc_score = np.array(jaccard(stemming, rem_kws, lines=lines, set=set))
    positions = np.array(get_positions_encoded(lines=lines, set=set))
    rough_length = np.array(list(get_comment_length(rough=True, set=set)))
    length = np.array(list(get_comment_length(rough=False, set=set)))
    types = np.array(get_type_encoded(set=set))
    link_tag = np.array(get_links_tag(set=set))


    #TEXTUAL
    comments = get_comments(set=set)
    tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, lowercase=False, sublinear_tf=True)
    dt_matrix = tfidf_vector.fit_transform(comments)
    if normalized:
        dt_matrix = normalize(dt_matrix, norm='l1', axis=0)

    for i in range(7):
        features = np.array([])
        if i != 0:
            new_feature = rough_length.reshape((rough_length.shape[0], 1))
            features = scale(np.hstack((features, new_feature))) if features.size else new_feature
        if i != 1:
            new_feature = length.reshape((length.shape[0], 1))
            features = scale(np.hstack((features, new_feature))) if features.size else new_feature
        if i != 2:
            features = scale(np.hstack((features, jacc_score.reshape((jacc_score.shape[0], 1)))))
        if i != 3:
            features = scale(np.hstack((features, types.reshape((types.shape[0], 1)))))
        if i != 4:
            features = scale(np.hstack((features, positions.reshape((positions.shape[0], 1)))))
        if i != 5:
            features = scale(np.hstack((features, link_tag.reshape((link_tag.shape[0], 1)))))
        if i != 6:
            features = sparse.hstack((dt_matrix, features))

        do_kfold(
            classifiers=[Classifier(AdaBoostClassifier())],
            labels=get_labels(set=set),
            features=features,
            folder=f"ablation{str(i)}",
            voting=False,
        )

    return 1


if __name__ == "__main__":
    ablation()
