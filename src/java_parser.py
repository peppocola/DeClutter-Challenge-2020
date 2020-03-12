import antlr4
from java.antlr_unit import Java8Parser, Java8Lexer
from src.url_utils import get_text_by_url
from src.csv_utils import link_line_type_extractor
import re
from src.keys import line, javadoc


def code_parser(code, comment_line, comment_type):
    code = get_line(code, comment_line, comment_type)
    lexer = Java8Lexer.Java8Lexer(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)  # Identifier
    parser = Java8Parser.Java8Parser(stream)
    tree = parser.compilationUnit()
    print(tree.toStringTree(recog=parser))


def get_line(code, comment_line, comment_type):
    code = code.splitlines()
    if comment_type == line:
        if not re.match(r"^[\t\s]*//.*", code[comment_line - 1]):   #if the line doesn't start as a comment, the comment refers to this line
            return code[comment_line - 1]
        i = 0
        while re.match(r"^[\t\s]*//.*", code[comment_line + i]):    #while the line starts as a comment, ignore it. I do this because they use multiple line comment to simulate a block
            i += 1
        while re.match(r"^[\t\s]*$", code[comment_line + i]):       #ignore blank lines
            i += 1
        if re.match(r"^[\t\s]*}[\t\s]*$", code[comment_line + i]):  #if this matches, probabily the comment refers to the line before
            i = -2
            while re.match(r"^[\t\s]*$", code[comment_line + i]):   #ignore blank lines before the comment
                i -= 1
        return code[comment_line + i]                               #comment refers to that
    else:
        if not re.match(r"^[\t\s]*/\*.*", code[comment_line - 1]):  #if the line doesn't start as a comment, the comment refers to this line
            return code[comment_line - 1]
        i = 0
        while re.match(r"^[\t\s]*\*.*", code[comment_line + i]):    #while the line starts as a comment, ignore it
            i += 1
        while re.match(r"^[\t\s]*$", code[comment_line + i]):       #ignore blank lines
            i += 1
        if comment_type == javadoc:
            while re.match(r"^[\t\s]*@.*", code[comment_line + i]): #ignore annotations
                i += 1
            while re.match(r"^[\t\s]*$", code[comment_line + i]):   #ignore blank lines again
                i += 1
        elif re.match(r"^[\t\s]*}[\t\s]*$", code[comment_line + i]):#if this matches, probabily the comment refers to the line before
            return code[comment_line - 2]
        return code[comment_line + i]


def get_positions():
    comment_type = 0
    text_link = 1
    comment_line = 2

    data = link_line_type_extractor()
    for row in data:
        print(row[comment_line], row[comment_type], row[text_link])
        code = get_text_by_url(row[text_link])
        line = get_line(code, row[comment_line], row[comment_type])
        print(line)
        input()


if __name__ == '__main__':
    # code = open('../testers/test.txt', 'r').read()
    # code_parser(code, 151, javadoc)
    get_positions()
