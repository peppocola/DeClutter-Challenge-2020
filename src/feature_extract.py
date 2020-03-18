import re
from src.csv_utils import comment_parser, tags_parser
from src.code_parser import get_code_words, word_extractor


def tag_for_comment():
    tags_dict = {}
    comments = comment_parser()
    for i in range(len(comments)):
        tags_dict[i] = []

    tags = tags_parser()
    for tag in tags:
        i = 0
        for comment in comments:
            if re.search(tag, comment):
                tags_dict[i].append(tag)
            i += 1
    return tags_dict


def get_comment_words(stemming=True, rem_keyws=True):
    comments = comment_parser()
    words = []
    for comment in comments:
        words.append(word_extractor(comment, stemming, rem_keyws))
    return words


def jaccard():
    code = get_code_words()
    comments = get_comment_words()
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
    comments = comment_parser()
    return [len(comment) for comment in comments]


if __name__ == '__main__':
    jaccard()

