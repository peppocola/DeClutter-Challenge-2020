import antlr4
from java.antlr_unit import Java8Parser, Java8Lexer
import time

def code_parser(path):
    start_time = time.time()
    code = open(path, 'r').read()
    lexer = Java8Lexer.Java8Lexer(antlr4.InputStream(code))
    stream = antlr4.CommonTokenStream(lexer) #Identifier
    parser = Java8Parser.Java8Parser(stream)
    tree = parser.compilationUnit()
    print(tree.toStringTree(recog=parser))
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    code_parser('../testers/test.txt')
