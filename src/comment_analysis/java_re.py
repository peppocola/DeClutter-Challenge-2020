import re


class Position:
    def __init__(self, re_, name_):
        self.name = name_
        self.re = re_

    def get_name(self):
        return self.name

    def get_re(self):
        return self.re

    def match_(self, line):
        return re.match(self.re, line)


# method_re = r"^(?:@[a-zA-Z]*|public|protected|private|abstract|static|final|strictfp|synchronized|native|\s)* +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *(\{?|[^;])"
method_re = r"^(?:@[a-zA-Z]*|public|protected|private|abstract|static|final|strictfp|synchronized|native|\s)*\s*[\w\<\>\[\], ]*\("
method_name = 'method'
class_re = r"^(?:@[a-zA-Z]*|public|protected|private|abstract|static|final|strictfp|synchronized|native|\s)*(class|interface|@interface)\s"  #check interface
class_name = 'class'
assignment_re = r"^[\s]*\w[\s\w[\]\*<,\?>\.]*="
assignment_name = 'assignment'
method_call_re = r"^[\s]*[\w.]+\("
method_call_name = 'method call'
return_re = r"^[\s]*return(\s|;)"
return_name = 'return'
cycle_re = r"^[\s]*(for|while)\s"
cycle_name = 'cycle'
if_re = r"^\s*(if|switch)(\s|\()"
if_name = 'if'
catch_re = r"^[\s]*}?[\s]*catch"
catch_name = 'catch'
variable_re = r"^(?:@[a-zA-Z]*|public|protected|private|abstract|static|final|strictfp|synchronized|native|\s)*\w+(<[a-zA-Z<>.\?]*>)?\s+\w+\s*;"
variable_name = 'variable'
enum_re = r"^(?:@[a-zA-Z]*|public|protected|private|abstract|static|final|strictfp|synchronized|native|\s)* enum\s"
enum_name = 'enum'
package_import_re = r"\s*(package|import)\s" #check package/import
package_import_name = 'package_import'
else_re = r"\s*}?\s*else\s*{?"
else_name = 'else'
loop_cond_re = "\s*(continue|break)(\s|;)"
loop_cond_name = 'loop_cond'
requires_re = r"\s*requires\s"
requires_name = 'requires'
#requires not handled


method_ = Position(method_re, method_name)
class_ = Position(class_re, class_name)
assignment_ = Position(assignment_re, assignment_name)
method_call_ = Position(method_call_re, method_call_name)
return_ = Position(return_re, return_name)
cycle_ = Position(cycle_re, cycle_name)
if_ = Position(if_re, if_name)
catch_ = Position(catch_re, catch_name)
variable_ = Position(variable_re, variable_name)
enum_ = Position(enum_re, enum_name)
package_import_ = Position(package_import_re, package_import_name)
else_ = Position(else_re, else_name)
loop_cond_ = Position(loop_cond_re, loop_cond_name)
requires_ = Position(requires_re, requires_name)


def get_position(focus_line):
    if cycle_.match_(focus_line):
        return cycle_.get_name()
    if if_.match_(focus_line):
        return if_.get_name()
    if else_.match_(focus_line):
        return else_.get_name()
    if catch_.match_(focus_line):
        return catch_.get_name()
    if return_.match_(focus_line):
        return return_.get_name()
    if method_call_.match_(focus_line):
        return method_call_.get_name()
    if method_.match_(focus_line):
        return method_.get_name()
    if enum_.match_(focus_line):
        return enum_.get_name()
    if class_.match_(focus_line):
        return class_.get_name()
    if assignment_.match_(focus_line):
        return assignment_.get_name()
    if variable_.match_(focus_line):
        return variable_.get_name()
    if package_import_.match_(focus_line):
        return package_import_.get_name()
    if loop_cond_.match_(focus_line):
        return loop_cond_.get_name()
    return requires_.get_name() if requires_.match_(focus_line) else 'error'
