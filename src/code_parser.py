from src.url_utils import get_text_by_url
from src.csv_utils import get_link_line_type, get_keywords
from src.keys import line
from nltk.stem.porter import *
from spacy.lang.en.stop_words import STOP_WORDS


def get_line(code, comment_line, comment_type):
    code = code.splitlines()
    try:
        if comment_type == line:
            if not re.match(r"^[\s]*//.*", code[comment_line - 1]):  # if the line doesn't start as a comment, the comment refers to this line
                if not re.match(r"^[\s]*}?[\s]*(else|try|finally)?[\s]*{?[\s]*//.*[\s]*$", code[comment_line - 1]):  # if the line isnt just brackets and some keywords, the foucs line is the comment_line
                    return code[comment_line - 1]
            i = 0
            while re.match(r"^[\s]*//.*", code[comment_line + i]) or re.match(r"^[\s]*$", code[comment_line + i]) or re.match(r"[\s]*[^}{](try|else|finally)[\s]*{?", code[comment_line + i]):  # while the line starts as a comment, ignore it. I do this because they use multiple line comment to simulate a block
                i += 1
            if re.match(r"^[\s]*}.*", code[comment_line + i]) or re.match(r"[\s]*(try|else|finally)[\s]*{?", code[comment_line + i]):  # if this matches, the block is empty so i take the first non comment non empty line before the comment.
                i = -2
                while re.match(r"^[\s]*//.*", code[comment_line + i]) or re.match(r"^[\s]*$", code[comment_line + i]) or re.match(r"^[\s]*/\*.*", code[comment_line + i]) or re.match(r"^\*", code[comment_line + i]) or re.match(r"^[\s]*\*/.*", code[comment_line + i]):  # while the line is a comment or is blank, ignore it
                    i -= 1
            return code[comment_line + i]  # comment refers to that
        # r"^[\s]*}?[\s]*(else|try|finally)?[\s]*{?[\s]*.*$"

        else:  # block or javadoc
            if not re.match(r"^[\s]*/\*.*", code[comment_line - 1]):  # if the line doesn't start as a comment, the comment refers to this line
                return code[comment_line - 1]
            if comment_line >= len(code) - 1:
                return code[comment_line - 2]
            i = 0
            if not re.match(r"^[\s]*.*\*/", code[comment_line - 1]):
                while not re.match(r"^[\s]*\*/", code[comment_line + i]):
                    i += 1
                i += 1
            while re.match(r"^[\s]*$", code[comment_line + i]) or re.match(r"^[\s]*@[^\s]*[\s]*$", code[comment_line + i]) or re.match(r"^[\s]*//.*", code[comment_line + i]) or re.match(r"[\s]*[^}{](try|else|finally)[\s]*{?", code[comment_line + i]):  # while the line starts as a comment or is blank, or is an annotation, ignore it
                i += 1
            if re.match(r"^[\s]*}[\s]*.*", code[comment_line + i]) or re.match(r"[\s]*(try|else|finally)[\s]*{?", code[comment_line + i]):  # if this matches, probabily the comment refers to the line before
                i = -2
                while re.match(r"^[\s]*//.*", code[comment_line + i]) or re.match(r"^[\s]*$",
                                                                                  code[comment_line + i]) or re.match(
                        r"^[\s]*/\*.*", code[comment_line + i]) or re.match(r"^\*", code[comment_line + i]) or re.match(
                        r"^[\s]*\*/.*", code[comment_line + i]):  # while the line is a comment or is blank, ignore it
                    i -= 1
            return code[comment_line + i]
    except IndexError:
        return ""


def get_positions():
    comment_type = 0
    text_link = 1
    comment_line = 2

    data = get_link_line_type()
    #random.shuffle(data)
    file = []
    for row in data:
        print(row[comment_line], row[comment_type], row[text_link]+"#L"+str(row[comment_line]))
        code = get_text_by_url(row[text_link])
        focus_line = get_line(code, row[comment_line], row[comment_type])
        print(focus_line)
        print(word_extractor(focus_line))

        # print(line_type_identifier(focus_line))
        # INCOMPLETE


def get_lines():
    comment_type = 0
    text_link = 1
    comment_line = 2

    data = get_link_line_type()
    lines = []
    for row in data:
        code = get_text_by_url(row[text_link])
        focus_line = get_line(code, row[comment_line], row[comment_type])
        lines.append(focus_line)
    return lines


def get_code_words(stemming=True, rem_keyws=True):
    lines = get_lines()
    words = []
    for line in lines:
        words.append(word_extractor(line, stemming, rem_keyws))
    return words


def word_extractor(string, stemming=True, rem_keyws=True):
    string = remove_line_comment(string)
    string = remove_block_comment(string)
    splitted = code_split(string)
    words = []
    for part in splitted:
        camel_case_parts = camel_case_split(part)
        for camel in camel_case_parts:
            words.append(camel.lower())
    if stemming and rem_keyws:
        return stem(remove_keywords(words))
    elif stemming:
        return stem(words)
    else:
        return remove_keywords(words)


def remove_keywords(words):
    keywords = get_keywords()
    non_keywords = []
    for word in words:
        if word not in keywords:
            non_keywords.append(word)
    return non_keywords


def stem(words):
    stemmer = PorterStemmer()
    stemmed = []
    for token in words:
        stemmed.append(stemmer.stem(token))
    return stemmed


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
            if not in_string:
                if maybe_block is True:
                    if block is True:
                        found = True
                        end_index = i
                    break
                else:
                    maybe_block = True
                    init_index = i
        i += 1
    if found is True:
        return string[:init_index] + string[end_index + 1:]
    return string


def code_split(string):
    words = re.split(r'\\n|\?|&|\\|;|,|\*|\(|\)|\{|\s|\.|/|_|:|=|<|>|\||!|"|\+|-|\[|\]|\'|\}|\^|#|%', string)
    words = list(filter(lambda a: a != "", words))
    return words


def remove_stopwords(tokens, language='English'):
    stop_words = STOP_WORDS
    relevant_words = []
    for token in tokens:
        if token not in stop_words:
            relevant_words.append(token)
    return relevant_words


def tokenizer(string, rem_stop=False, stemming=False, rem_kws=False):
    tokens = code_split(string)
    new_tokens = []
    for token in tokens:
        for t in camel_case_split(token):
            new_tokens.append(t.lower())
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
    # get_positions()
    # line_type_identifier("ciao")
    # code_parser3()
    # print(word_extractor("ciao mamma /*css rff*/"))
    print(tokenizer("tuaMadre@QuellaTroia @param"))
    print(camel_case_split("tuaMadre@QuellaTroia @param"))
    print(code_split("tuaMadre@QuellaTroia @param"))