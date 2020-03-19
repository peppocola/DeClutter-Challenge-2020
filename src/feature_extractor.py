import re
from sklearn.feature_extraction.text import TfidfVectorizer
from src.csv_utils import get_comments, get_tags, get_javadoc_comments
from src.code_parser import get_code_words, word_extractor, tokenizer
from src.keys import reports_outpath


def get_tfidf_features():
    comments = get_comments()
    tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, lowercase=False)
    tfidf_vector.fit_transform(comments)
    file = open(reports_outpath+"tfidf_features.txt", 'w')
    for key in tfidf_vector.vocabulary_.keys():
        file.write(key)
        file.write("\n")
    return tfidf_vector.vocabulary_


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


def jaccard(stemming=True, rem_keyws=True):
    code = get_code_words(stemming, rem_keyws)
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
    for list in tags:
        found = False
        for tag in list:
            if tag == "{@link.*}":
                found = True
                bool_vector.append(has_link)
                break
        if not found:
            bool_vector.append(not has_link)
    return bool_vector



if __name__ == '__main__':
    #jaccard()
    #print(get_javadoc_tags())
    #print(get_tfidf_features())
    print(get_tag_for_comment())
    print(get_links_tag())
