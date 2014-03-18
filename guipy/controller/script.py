from controller.source import SourceEditController
from PyQt4 import QtGui
from model.script import ScriptModel
from utils.gui import defaultFont

try:
    from pyeditor import PyEdit
    hasPyCode = True
except ImportError as e:
    hasPyCode = False 

    #sys.path.append("./pycodelocal/syntaxhighlighter")
try:
        from external.highlighter import SyntaxHighlighter, load_syntax
        from external.highlighter.python27 import syntax
        scheme = {
                "syntax_comment": dict(color="green", italic=True),
                "syntax_string": "magenta",
                "syntax_builtin": "red",
                "syntax_keyword": ("darkred", True),
                "syntax_number": "blue",
            }
except ImportError:
        SyntaxHighlighter = None
    


class ScriptController(SourceEditController):
    
    def __init__(self, document, model = ScriptModel()):
        SourceEditController.__init__(self, document, model)

    def create_source_editor(self, parent = None):
        edit = QtGui.QPlainTextEdit(parent)
        edit.setFont(defaultFont)
        if hasPyCode:
            self.pyedit = PyEdit(".", edit)
        if SyntaxHighlighter:
            parts_scanner, code_scanner, formats = load_syntax(syntax, scheme)
            self.highlighter = SyntaxHighlighter(edit.document(), parts_scanner, code_scanner, formats, default_font=defaultFont)
        edit.setReadOnly(self.model.is_read_only())
        return edit
    
    def on_edit_enter(self):
        super(ScriptController, self).on_edit_enter()
        self.pyedit.prefix = self.document.stubs()