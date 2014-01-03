from models.base import SectionModelTreeBased
from PyQt4 import QtGui

import sys
try:
    from pyeditor import PyEdit
    from pycode import pyqtfrontend
    hasPyCode = True
except ImportError:
    hasPyCode = False 

if hasPyCode:
    sys.path.append("syntaxhighlighter")
    try:
        from highlighter import SyntaxHighlighter, load_syntax
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

class ScriptModel(SectionModelTreeBased):
    
    def __init__(self):
        SectionModelTreeBased.__init__(self, 'script')

    def createSourceEditor(self):
        if hasPyCode:
            edit = QtGui.QPlainTextEdit(None)
            self.pyedit = PyEdit(".", edit, prefix="from PyQt4 import QtGui")
            font = QtGui.QFont()    #!! beg - move to get?
            font.setFamily("Courier")
            font.setPointSize(10)
            edit.setFont(font)
            if SyntaxHighlighter:
                parts_scanner, code_scanner, formats = load_syntax(syntax, scheme)
                self.highlighter = SyntaxHighlighter(edit.document(), parts_scanner, code_scanner, formats, default_font=font)    #!! end
            return edit
        else:
            return QtGui.QPlainTextEdit(None)