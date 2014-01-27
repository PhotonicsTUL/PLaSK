from controler.source import SourceEditControler
from PyQt4 import QtGui
from model.script import ScriptModel
from utils import defaultFont

import sys
try:
    from pyeditor import PyEdit
    hasPyCode = True
except ImportError:
    hasPyCode = False 

if hasPyCode:
    sys.path.append("syntaxhighlighter")
    try:
        from pycodelocal.highlighter import SyntaxHighlighter, load_syntax
        from highlighter.python27 import syntax
    except ImportError:
        SyntaxHighlighter = None
    
    scheme = {
        "syntax_comment": dict(color="green", italic=True),
        "syntax_string": "magenta",
        "syntax_builtin": "red",
        "syntax_keyword": ("darkred", True),
        "syntax_number": "blue",
    }

class ScriptControler(SourceEditControler):
    
    def __init__(self, document, model = ScriptModel()):
        SourceEditControler.__init__(self, document, model)

    def createSourceEditor(self, parent = None):
        if hasPyCode:
            edit = QtGui.QPlainTextEdit(parent)
            edit.setFont(defaultFont)
            self.pyedit = PyEdit(".", edit, prefix="from PyQt4 import QtGui")
            if SyntaxHighlighter:
                parts_scanner, code_scanner, formats = load_syntax(syntax, scheme)
                self.highlighter = SyntaxHighlighter(edit.document(), parts_scanner, code_scanner, formats, default_font=defaultFont)
        else:
            edit = QtGui.QPlainTextEdit(parent)
            edit.setFont(defaultFont)
        edit.setReadOnly(self.model.isReadOnly())
        return edit