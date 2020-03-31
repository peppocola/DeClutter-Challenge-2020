import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, f_classif, RFECV, mutual_info_classif, SelectKBest
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.preprocessing import LabelEncoder

from src.csv_utils import get_comments, get_tags, get_javadoc_comments, get_labels, get_type
from src.code_parser import get_code_words, word_extractor, tokenizer
from src.keys import reports_outpath, javadoc, line, block

import numpy as np


def get_k_best_features_sfs(classifiers, k=(1, 500), forward=True, scoring='f1_macro'):
    X = get_comments()
    y = get_labels()

    for classifier in classifiers:
        clf = classifier()

        sfs1 = sfs(clf,
                   k_features=k,
                   forward=forward,
                   floating=False,
                   verbose=2,
                   scoring=scoring,
                   cv=5)

        pipeline = Pipeline(
            [('tfidf', TfidfVectorizer(tokenizer=tokenizer, lowercase=False)), ('sfs_feature_selection', sfs1),
             ('clf', clf)])
        pipeline.fit(X, y)
        # y_pred = pipeline.predict(X_dev)
        support = pipeline.named_steps['sfs_feature_selection'].support_
        feature_names = pipeline.named_steps['tfidf'].get_feature_names()
        print(np.array(feature_names)[support])


def get_k_best_features(scoring, k=20):
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, lowercase=False)
    X = vectorizer.fit_transform(get_comments())

    scores = scoring(X, get_labels())[0]
    wscores = zip(vectorizer.get_feature_names(), scores)
    sorted_scores = sorted(wscores, key=lambda x: x[1])
    top_k = zip(*sorted_scores[-k:])
    return list(top_k)


def get_MI_k_best_features(k=20):
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, lowercase=False)
    X = vectorizer.fit_transform(get_comments())
    select = SelectKBest(mutual_info_classif, k)
    X_new = select.fit_transform(X, get_labels())

    feature_names = vectorizer.get_feature_names()
    return np.array(feature_names)[select.get_support()]


def get_chi2_k_best_features(k=20):
    return get_k_best_features(chi2, k)


def get_fclassif_best_features(k=20):
    return get_k_best_features(f_classif, k)


def get_k_best_features_rfe(classifiers):
    X = get_comments()
    y = get_labels()
    out_dict = {}
    for classifier in classifiers:
        rfe = RFECV(estimator=classifier(), step=1, scoring='f1_macro')

        pipeline = Pipeline(
            [('tfidf', TfidfVectorizer(tokenizer=tokenizer, lowercase=False)), ('rfe_feature_selection', rfe),
             ('clf', classifier())])
        pipeline.fit(X, y)
        # y_pred = pipeline.predict(X_dev)
        support = pipeline.named_steps['rfe_feature_selection'].support_
        feature_names = pipeline.named_steps['tfidf'].get_feature_names()
        out_dict[classifier.__name__] = np.array(feature_names)[support]
    return out_dict


def get_tfidf_features(max_features=None):
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


def get_top_n_tfidf_features(top_n=50):
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


def get_tag_for_comment():
    return get_tag_for_list(get_comments())


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


def get_comment_words(stemming=True, rem_keyws=True):
    comments = get_comments()
    words = []
    for comment in comments:
        words.append(word_extractor(comment, stemming, rem_keyws))
    return words


def jaccard(stemming=True, rem_keyws=True, lines=None):
    if lines is None:
        code = get_code_words(stemming, rem_keyws)
    else:
        code = code = get_code_words(stemming, rem_keyws, lines)
    comments = get_comment_words(stemming, rem_keyws)
    score = []
    for i in range(len(comments)):
        score.append(get_jaccard_sim(code[i], comments[i]))
    return score


def get_jaccard_sim(first, second):
    a = set(first)
    b = set(second)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_comment_length():
    comments = get_comments()
    return [len(comment) for comment in comments]


def get_javadoc_tags():
    return get_tag_for_list(get_javadoc_comments())


def get_links_tag():
    has_link = True

    tags = get_tag_for_comment()
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


def normalize(nparray):
   return (nparray - nparray.mean(axis=0)) / nparray.std(axis=0)


def get_type_encoded():
    types = get_type()
    le = LabelEncoder()
    return le.fit_transform(types)


if __name__ == '__main__':
    # jaccard()
    # print(get_javadoc_tags())
    # print(get_top_n_tfidf_features(50))
    # print(get_chi2_k_best_features())
    # print(get_fclassif_best_features())
    dicty = get_k_best_features_rfe([RandomForestClassifier])
    print(dicty)
    # print(len(dicty[LogisticRegression.__name__]))
    # get_k_best_features_sfs([LogisticRegression], forward=True)
    # get_k_best_features_sfs([LogisticRegression], forward=False)
    # print(get_MI_k_best_features(k=50))
    #print(get_type_encoded())
