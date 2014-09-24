# This code was taken from ropeide - a Python refactoring IDE
# by Ali Gholami Rudi <aligrudi at users sourceforge net>
# License: GPL
from rope.base import ast


class PythonASTOutlineNode(object):

    def __init__(self, ast_node):
        self.name = ast_node.name
        self.node = ast_node
        self.children = None

    def get_name(self):
        return self.name

    def get_line_number(self):
        return self.node.lineno

    def get_children(self):
        if self.children is None:
            self.children = _get_ast_children(self.node)
        return self.children

    def get_kind(self):
        """Return the type of this node

        Returns 'function' for functions can 'class' for classes.

        """
        if isinstance(self.node, ast.FunctionDef):
            return 'function'
        if isinstance(self.node, ast.ClassDef):
            return 'class'

    def __cmp__(self, obj):
        return cmp(self.get_line_number(), obj.get_line_number())


class _ASTDefinedVisitor(object):

    def __init__(self):
        self.result = []

    def _FunctionDef(self, node):
        self.result.append(PythonASTOutlineNode(node))

    def _ClassDef(self, node):
        self.result.append(PythonASTOutlineNode(node))


def _get_ast_children(node):
    visitor = _ASTDefinedVisitor()
    for child in ast.get_child_nodes(node):
        ast.walk(child, visitor)
    return visitor.result


def outline(project, source_code):
    if isinstance(source_code, unicode):
        source_code = source_code.encode('utf-8')
    ast_node = ast.parse(source_code)
    return _get_ast_children(ast_node)
