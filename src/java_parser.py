import antlr4
import javalang
from java.antlr_unit import Java8Parser, Java8Lexer
from src.url_utils import get_text_by_url
from src.csv_utils import link_line_type_extractor
import re
import inspect
import random
from src.keys import line, javadoc


def code_parser3():
    code = " public static void showContextMenu(TextArea textArea, ContextMenu contextMenu, ContextMenuEvent e) {"
    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    members = inspect.getmembers(parser, predicate=inspect.ismethod)
    for i in members:
        try:
            parser.i[0]()
            ast = parser.parse_member_declaration()
            if type(ast) is javalang.tree.MethodInvocation:
                print(ast)
        except javalang.parser.JavaSyntaxError as err:
            print("wrong syntax", err)


def code_parser2():
    code = " public static void showContextMenu(TextArea textArea, ContextMenu contextMenu, ContextMenuEvent e) {"
    tree = javalang.parse.parse(code)
    return tree.types


def is_class_declaration(code):
    lexer = Java8Lexer.Java8Lexer(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = Java8Parser.Java8Parser(stream)
    tree = parser.classDeclaration()
    print(tree.toStringTree(recog=parser))


def is_method_declaration(code):
    lexer = Java8Lexer.Java8Lexer(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = Java8Parser.Java8Parser(stream)
    tree = parser.methodDeclaration()
    print(tree.toStringTree(recog=parser))


def is_statement(code):
    lexer = Java8Lexer.Java8Lexer(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = Java8Parser.Java8Parser(stream)
    tree = parser.statement()
    print(tree.toStringTree(recog=parser))


def is_local_varable_declaration(code):
    lexer = Java8Lexer.Java8Lexer(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = Java8Parser.Java8Parser(stream)
    tree = parser.localVariableDeclaration()
    print(tree.toStringTree(recog=parser))


def is_if(code):
    lexer = Java8Lexer.Java8Lexer(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = Java8Parser.Java8Parser(stream)
    tree = parser.ifThenStatement()
    print(tree.toStringTree(recog=parser))


def line_type_identifier(code):
    code = "public static void showContextMenu(TextArea textArea, ContextMenu contextMenu, ContextMenuEvent e) {"
    is_class_declaration(code)
    is_method_declaration(code)
    is_local_varable_declaration(code)
    is_statement(code)

    input()
    code = "TABLE_ICONS.put(SpecialField.PRINTED, icon);"
    is_class_declaration(code)
    is_method_declaration(code)
    is_local_varable_declaration(code)
    is_statement(code)

    input()
    code = "public static class{"
    is_class_declaration(code)
    is_method_declaration(code)
    is_local_varable_declaration(code)
    is_statement(code)

    code = "if (identicalFields.contains(field)) {"
    is_class_declaration(code)
    is_method_declaration(code)
    is_local_varable_declaration(code)
    is_statement(code)
    is_if(code)


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

    data = link_line_type_extractor()
    random.shuffle(data)
    file = []
    for row in data:
        print(row[comment_line], row[comment_type], row[text_link]+"#L"+str(row[comment_line]))
        code = get_text_by_url(row[text_link])
        focus_line = get_line(code, row[comment_line], row[comment_type])
        print(focus_line)
        print(word_extractor(focus_line))

        # print(line_type_identifier(focus_line))


def word_extractor(string):
    string = remove_line_comment(remove_block_comment(string))
    splitted = first_split(string)
    words = []
    for part in splitted:
        camel_case_parts = camel_case_split(part)
        for camel in camel_case_parts:
            words.append(camel.lower())
    return words  # TODO:lemma and find synset for jaccard


def camel_case_split(string):
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
                break
        elif comment is True:
            i += 1
            comment = False
        elif escape is True:
            escape = False
        if comment is False:
            i += 1
    return string[:i]


def remove_block_comment(string): #somehow not working properly
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


def first_split(string):
    words = re.split(r'\\n|&|\\|;|,|\*|\(|\)|\{|\s|\.|/|@|_|:|=|<|>|\||!|"|\+|-|\[|\]|\'|\|\?|\}|\^', string) #?not splitting
    words = list(filter(lambda a: a != "", words))
    return words


if __name__ == '__main__':
    # code = open('../testers/test.txt', 'r').read()
    # code_parser(code, 151, javadoc)
    get_positions()
    # line_type_identifier("ciao")
    # code_parser3()
