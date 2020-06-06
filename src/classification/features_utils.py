import re

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2, f_classif, RFECV, mutual_info_classif, SelectKBest
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.preprocessing import LabelEncoder, normalize, scale

from src.csv.csv_utils import get_comments, get_tags, get_javadoc_comments, get_labels, get_type, write_csv
from src.comment_analysis.parsing_utils import get_code_words, word_extractor, tokenizer, get_lines, get_positions_encoded
from src.keys import reports_outpath

import numpy as np


def get_features(set='train', stemming=True, rem_kws=True, scaled=True, lines=None):
    if lines is None:
        lines = get_lines(set=set)
    jacc_score = np.array(jaccard(stemming, rem_kws, lines=lines, set=set))
    positions = np.array(get_positions_encoded(lines=lines, set=set))
    rough_length = np.array([x for x in get_comment_length(rough=True, set=set)])
    length = np.array([x for x in get_comment_length(rough=False, set=set)])
    types = np.array(get_type_encoded(set=set))
    link_tag = np.array(get_links_tag(set=set))

    if scaled:
        features = scale(rough_length.reshape((rough_length.shape[0], 1)))
        features = scale(np.hstack((features, length.reshape((length.shape[0], 1)))))
        features = scale(np.hstack((features, jacc_score.reshape((jacc_score.shape[0], 1)))))
        features = scale(np.hstack((features, types.reshape((types.shape[0], 1)))))
        features = scale(np.hstack((features, positions.reshape((positions.shape[0], 1)))))
        features = scale(np.hstack((features, link_tag.reshape((link_tag.shape[0], 1)))))
    else:
        features = rough_length.reshape((rough_length.shape[0], 1))
        features = np.hstack((features, length.reshape((length.shape[0], 1))))
        features = np.hstack((features, jacc_score.reshape((jacc_score.shape[0], 1))))
        features = np.hstack((features, types.reshape((types.shape[0], 1))))
        features = np.hstack((features, positions.reshape((positions.shape[0], 1))))
        features = np.hstack((features, link_tag.reshape((link_tag.shape[0], 1))))

    return features


def get_tfidf_features(set='train', normalized=True):
    comments = get_comments(set=set)
    tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, lowercase=False, sublinear_tf=True)
    dt_matrix = tfidf_vector.fit_transform(comments)
    if normalized:
        dt_matrix = normalize(dt_matrix, norm='l1', axis=0)
    return dt_matrix


def get_wc_features(set='train', normalized=True):
    comments = get_comments(set=set)
    wc_vector = CountVectorizer(tokenizer=tokenizer, lowercase=False)
    dt_matrix = wc_vector.fit_transform(comments)
    if normalized:
        dt_matrix = normalize(dt_matrix,  norm='l1', axis=0)
    return dt_matrix


def get_both_features(set='train', scaled=True, normalized=True, stemming=True, rem_kws=True, lines=None, tf_idf=True):
    if tf_idf:
        dt_matrix = get_tfidf_features(set=set, normalized=normalized)
    else:
        dt_matrix = get_wc_features(set=set, normalized=normalized)
    features = get_features(set=set, scaled=scaled, stemming=stemming, rem_kws=rem_kws, lines=lines)
    return sparse.hstack((dt_matrix, features))


def do_feature_selection(method, classifier, feature_type='tfidf'):
    y = get_labels()

    clf = classifier()
    pipeline = Pipeline(
        [('feature_selection', method), ('clf', clf)])
    if feature_type == 'tfidf':
        features, feature_names = get_tfidf_feature_names()
        pipeline.fit(features, y)
    elif feature_type == 'nontextual':
        features, feature_names = get_nontextual_features()
        pipeline.fit(np.array(features), y)
    else:
        raise ValueError

    if isinstance(method, sfs):
        if feature_type == 'nontextual':
            selected = []
            for f in pipeline.named_steps['feature_selection'].k_feature_names_:
                selected.append(feature_names[int(f)])
            return selected
        else:
            return pipeline.named_steps['feature_selection'].k_feature_names_
    else:
        support = pipeline.named_steps['feature_selection'].support_
        return np.array(feature_names)[support]


def get_best_sfs_features(classifier, k=1, forward=True, scoring='f1_macro', feature_type='tfidf'):
    sfs_ = sfs(estimator=classifier(), k_features=k, forward=forward, floating=False, verbose=2, scoring=scoring, cv=10)
    return do_feature_selection(sfs_, classifier, feature_type)


def get_best_rfe_features(classifier, feature_type='tfidf'):
    rfe = RFECV(estimator=classifier(), step=1, scoring='f1_macro')
    return do_feature_selection(rfe, classifier, feature_type)


def get_best_tfidf_features(scoring, k=20):
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, lowercase=False)
    X = vectorizer.fit_transform(get_comments())
    return get_best_features(scoring, features=X, feature_names=vectorizer.get_feature_names(), k=k)


def get_best_features(scoring, features, feature_names, k='all', csv=False):
    select = SelectKBest(scoring, k)
    X_new = select.fit_transform(features, get_labels())
    if csv:
        write_csv(['feature', 'score'], [feature_names, select.scores_], scoring.__name__ + '_features')
    return np.array(feature_names)[select.get_support()]


def get_nontextual_features():
    lines = get_lines()
    positions = get_positions_encoded(lines=lines)
    jacc_score = np.array(jaccard(lines=lines))
    length = np.array([x / 100 for x in get_comment_length()])
    types = np.array(get_type_encoded())

    features = []
    for i in range(len(length)):
        features.append([jacc_score[i], length[i], types[i], positions[i]])
    feature_names = ['jacc_score', 'length', 'types', 'positions']
    return features, feature_names


def get_mi_best_features():
    features = get_nontextual_features()
    return mutual_info_classif(features, get_labels())


def get_mi_tfidf_best_features(k=20):
    features, feature_names = get_tfidf_feature_names()
    return get_best_features(scoring=mutual_info_classif, features=features, feature_names=feature_names, k=k, csv=True)


def get_chi2_tfidf_best_features(k=20):
    features, feature_names = get_tfidf_feature_names()
    return get_best_features(scoring=chi2, features=features, feature_names=feature_names, k=k, csv=True)


def get_fclassif_tfidf_best_features(k=20):
    features, feature_names = get_tfidf_feature_names()
    return get_best_features(scoring=f_classif, features=features, feature_names=feature_names, k=k, csv=True)


def get_tfidf_feature_names():
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, lowercase=False)
    features = vectorizer.fit_transform(get_comments())
    feature_names = vectorizer.get_feature_names()
    return features, feature_names


def get_tfidf_vocabulary(max_features=None):
    comments = get_comments()
    if max_features is None:
        tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, lowercase=False)
    else:
        tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, lowercase=False, max_features=max_features)
    tfidf_vector.fit_transform(comments)
    file = open(reports_outpath + "tfidf_features.txt", 'w')
    for key in tfidf_vector.vocabulary_.keys():
        file.write(key)
        file.write("\n")
    return tfidf_vector.vocabulary_


def get_top_tfidf_features(top_n=50):
    comments = get_comments()
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, lowercase=False)
    X = vectorizer.fit_transform(comments)
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()
    top_features = [features[i] for i in indices[:top_n]]

    file = open(reports_outpath + "top_" + str(top_n) + "_tfidf_features.txt", 'w')
    for feature in top_features:
        file.write(feature)
        file.write("\n")
    return top_features


def get_tag_for_comment(set='train'):
    return get_tag_for_list(get_comments(set))


def get_tag_for_list(comments):
    tags_list = []
    for i in range(len(comments)):
        tags_list.append([])

    tags = get_tags()
    for tag in tags:
        i = 0
        for comment in comments:
            if re.search(tag, comment):
                tags_list[i].append(tag)
            i += 1
    return tags_list


def get_comment_words(stemming=True, rem_keyws=True, set='train'):
    comments = get_comments(set)
    words = []
    for comment in comments:
        words.append(word_extractor(comment, stemming, rem_keyws))
    return words


def jaccard(stemming=True, rem_keyws=True, lines=None, set='train'):
    if lines is None:
        code = get_code_words(stemming, rem_keyws, set=set)
    else:
        code = get_code_words(stemming, rem_keyws, lines, set=set)
    comments = get_comment_words(stemming, rem_keyws, set=set)
    score = []
    for i in range(len(comments)):
        score.append(get_jaccard_sim(code[i], comments[i]))
    return score


def get_jaccard_sim(first, second):
    a = set(first)
    b = set(second)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_comment_length(set='train', rough=True):
    comments = get_comments(set=set)
    if rough:
        return [len(comment) for comment in comments]
    else:
        return [len(comment.split()) for comment in comments]


def get_javadoc_tags():
    return get_tag_for_list(get_javadoc_comments())


def get_links_tag(set='train'):
    has_link = True

    tags = get_tag_for_comment(set=set)
    bool_vector = []
    for list_ in tags:
        found = False
        for tag in list_:
            if tag == "{@link.*}":
                found = True
                bool_vector.append(has_link)
                break
        if not found:
            bool_vector.append(not has_link)
    return bool_vector


def get_type_encoded(set='train'):
    types = get_type(set=set)
    le = LabelEncoder()
    return le.fit_transform(types)


def get_no_sep(set='train'):
    comments = get_comments(set=set)
    return [count_sep(x) for x in comments]


def count_sep(string):
    matches = re.findall(r'\\n|\?|&|\\|;|,|\*|\(|\)|\{|\.|/|_|:|=|<|>|\||!|"|\+|-|\[|\]|\'|\}|\^|#|%', string)
    return len(matches)


if __name__ == '__main__':
    # print(get_mi_tfidf_best_features())
    # print(get_chi2_tfidf_best_features())
    # print(get_fclassif_tfidf_best_features())
    # print(get_best_rfe_features(AdaBoostClassifier, feature_type='nontextual'))
    # print(get_best_sfs_features(forward=True, classifier=RandomForestClassifier, feature_type='tfidf', k=(1,500)))
    print([x for x in zip(get_no_sep(), get_comments())])
