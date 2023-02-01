import json

from sklearn.preprocessing import LabelEncoder

from src.comment_analysis.url_utils import get_text_by_url
from src.csv.csv_utils import get_link_line_type, get_keywords
from src.keys import line, serialize_outpath
from nltk.stem.porter import *
from spacy.lang.en.stop_words import STOP_WORDS
from src.comment_analysis.java_re import *


def get_line(code, comment_line, comment_type):
    code = code.splitlines()
    try:
        if comment_type == line:
            if not re.match(
                r"^[\s]*//.*", code[comment_line - 1]
            ) and not re.match(
                r"^[\s]*}?[\s]*(else|try|finally)?[\s]*{?[\s]*//.*[\s]*$",
                code[comment_line - 1],
            ):
                return code[comment_line - 1]
            i = 0
            while re.match(r"^[\s]*//.*", code[comment_line + i]) or re.match(r"^[\s]*$", code[comment_line + i]) or re.match(r"[\s]*[^}{](try|else|finally)[\s]*{?", code[comment_line + i]):  # while the line starts as a comment, ignore it. I do this because they use multiple line comment to simulate a block
                i += 1
            if re.match(r"^[\s]*}.*", code[comment_line + i]) or re.match(r"[\s]*(try|else|finally)[\s]*{?", code[
                comment_line + i]):  # if this matches, the block is empty so i take the first non comment non empty line before the comment.
                i = -2
                while re.match(r"^[\s]*//.*", code[comment_line + i]) or re.match(r"^[\s]*$",
                                                                                  code[comment_line + i]) or re.match(
                        r"^[\s]*/\*.*", code[comment_line + i]) or re.match(r"^\*", code[comment_line + i]) or re.match(
                        r"^[\s]*\*/.*", code[comment_line + i]):  # while the line is a comment or is blank, ignore it
                    i -= 1
        else:  # block or javadoc
            # if the line doesn't start as a comment, the comment refers to this line
            if not re.match(r"^[\s]*/\*.*", code[comment_line - 1]):
                return code[comment_line - 1]
            if comment_line >= len(code) - 1:
                return code[comment_line - 2]
            i = 0
            if not re.match(r"^[\s]*.*\*/", code[comment_line - 1]):
                while not re.match(r"^[\s]*\*/", code[comment_line + i]):
                    i += 1
                i += 1
            # while the line starts as a comment or is blank, or is an annotation, ignore it
            while re.match(r"^[\s]*$", code[comment_line + i]) or re.match(r"^[\s]*@[^\s]*[\s]*$", code[comment_line + i]) or re.match(r"^[\s]*//.*", code[comment_line + i]) or re.match(r"[\s]*[^}{](try|else|finally)[\s]*{?", code[comment_line + i]):
                i += 1
            # if this matches, probabily the comment refers to the line before
            if re.match(r"^[\s]*}[\s]*.*", code[comment_line + i]) or re.match(r"[\s]*(try|else|finally)[\s]*{?", code[comment_line + i]):
                i = -2
                # while the line is a comment or is blank, ignore it
                while re.match(r"^[\s]*//.*", code[comment_line + i]) or re.match(r"^[\s]*$", code[comment_line + i]) or re.match(r"^[\s]*/\*.*", code[comment_line + i]) or re.match(r"^\*", code[comment_line + i]) or re.match(r"^[\s]*\*/.*", code[comment_line + i]):
                    i -= 1
        return code[comment_line + i]  # comment refers to that
    except IndexError:
        return ""


def get_positions(lines=None, set='train'):
    comment_type = 0
    text_link = 1
    comment_line = 2

    positions = []
    data = get_link_line_type(set=set)
    if lines is None:
        lines = get_lines(set=set)
    for i, _ in enumerate(data):
        #print(row[comment_line], row[comment_type], row[text_link] + "#L" + str(row[comment_line]))
        focus_line = lines[i]
        #print(focus_line)
        p = get_position(focus_line)
        positions.append(p)
    return positions


def get_positions_encoded(lines=None, set='train'):
    if lines is None:
        positions = get_positions(set=set)
    else:
        positions = get_positions(lines, set=set)
    le = LabelEncoder()
    return le.fit_transform(positions)


def get_lines(serialized=True, serialize=False, set='train'):
    if serialized:
        x = open(f'{serialize_outpath}serialized_{set}.json', 'r').read()
        return json.loads(x)
    comment_type = 0
    text_link = 1
    comment_line = 2

    data = get_link_line_type(set=set)
    lines = []
    for row in data:
        code = get_text_by_url(row[text_link])
        focus_line = get_line(code, row[comment_line], row[comment_type])
        lines.append(focus_line)
    if serialize:
        x = open(f'{serialize_outpath}serialized_{set}.json', 'w')
        x.write(json.dumps(lines))
    return lines


def get_code_words(stemming=True, rem_keyws=True, lines=None, set='train'):
    if lines is None:
        lines = get_lines(set=set, serialized=False, serialize=True)
    return [word_extractor(line, stemming, rem_keyws) for line in lines]


def word_extractor(string, stemming=True, rem_keyws=True):
    string = remove_line_comment(string)
    string = remove_block_comment(string)
    splitted = code_split(string)
    words = []
    for part in splitted:
        camel_case_parts = camel_case_split(part)
        words.extend(camel.lower() for camel in camel_case_parts)
    if stemming and rem_keyws:
        return stem(remove_keywords(words))
    elif stemming:
        return stem(words)
    else:
        return remove_keywords(words)


def remove_keywords(words):
    keywords = get_keywords()
    return [word for word in words if word not in keywords]


def stem(words):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in words]


def camel_case_split(string):
    if not string:
        return string
    words = [[string[0]]]

    for c in string[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)

    return [''.join(word) for word in words]


def remove_line_comment(string):
    in_string = False
    escape = False
    comment = False
    i = 0
    for char in string:
        if char == '"':
            if in_string is True:
                if escape is False:
                    in_string = False
                else:
                    escape = False
            else:
                in_string = True
        elif char == '\\':
            if in_string is True:
                escape = True
        elif char == '/':
            if comment is False:
                comment = True
            else:
                return string[:i]
        elif comment is True:
            i += 1
            comment = False
        elif escape is True:
            escape = False
        if comment is False:
            i += 1
    return string


def remove_block_comment(string):
    in_string = False
    escape = False
    block = False
    maybe_block = False
    found = False
    init_index = 0
    end_index = 0
    i = 0
    for char in string:
        if char == '*':
            if not in_string:
                if maybe_block is False:
                    if block is True:
                        maybe_block = True
                else:
                    block = True
        elif char == '/':
            if not in_string:
                if maybe_block is True:
                    if block is True:
                        found = True
                        end_index = i
                    break
                else:
                    maybe_block = True
                    init_index = i
        elif char == '\\':
            if in_string is True:
                escape = True
        elif char == '"':
            if in_string is True:
                if escape is False:
                    in_string = False
                else:
                    escape = False
            else:
                in_string = True
        i += 1
    if found is True:
        return string[:init_index] + string[end_index + 1:]
    return string


def code_split(string):
    words = re.split(r'\\n|\?|&|\\|;|,|\*|\(|\)|\{|\s|\.|/|_|:|=|<|>|\||!|"|\+|-|\[|\]|\'|\}|\^|#|%', string)
    words = list(filter(lambda a: a != "", words))
    return words


def remove_stopwords(tokens):
    stop_words = STOP_WORDS
    return [token for token in tokens if token not in stop_words]


def tokenizer(string, rem_stop=False, stemming=False, rem_kws=False):
    tokens = code_split(string)
    new_tokens = []
    for token in tokens:
        new_tokens.extend(t.lower() for t in camel_case_split(token))
    if rem_stop:
        new_tokens = remove_stopwords(new_tokens)
    if rem_kws:
        new_tokens = remove_keywords(new_tokens)
    if stemming:
        new_tokens = stem(new_tokens)
    return new_tokens


if __name__ == '__main__':
    # code = open('../testers/test.txt', 'r').read()
    # code_parser(code, 151, javadoc)
    print(get_lines(serialized=False, serialize=True))
    print('first')
    print(get_lines(serialized=True, serialize=False))
    # get_positions()
    # line_type_identifier("ciao")
    # code_parser3()
    # print(word_extractor("ciao mamma /*css rff*/"))
    # print(tokenizer("tuaMadre@QuellaTroia @param"))
    # print(camel_case_split("tuaMadre@QuellaTroia @param"))
    # print(code_split("tuaMadre@QuellaTroia @param"))
