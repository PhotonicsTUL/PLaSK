from ..qt import QtGui

from ..model.script import ScriptModel
from ..utils.gui import DEFAULT_FONT
from .source import SourceEditController
from ..utils.config import CONFIG

try:
    from ..pyeditor import PyEdit
    hasPyCode = True
except ImportError as e:
    hasPyCode = False
    #sys.path.append("./pycodelocal/syntaxhighlighter")

try:
    from ..external.highlighter import SyntaxHighlighter, load_syntax
    from ..external.highlighter.python27 import syntax

    def _parse_config(string):
        result = {}
        for item in string.split(','):
            item = item.strip()
            key, val = item.split('=')
            if val.lower() in ('true', 'yes'): val = True
            elif val.lower() in ('false', 'no'): val = False
            result[key] = val
        return result

    scheme = {
        "syntax_comment": _parse_config(CONFIG('syntax/python_comment', "color=green, italic=true")),
        "syntax_string": _parse_config(CONFIG('syntax/python_string', "color=blue")),
        "syntax_builtin": _parse_config(CONFIG('syntax/python_builtin', "color=red")),
        "syntax_keyword": _parse_config(CONFIG('syntax/python_keyword', "color=black, bold=true")),
        "syntax_number": _parse_config(CONFIG('syntax/python_number', "color=darkblue")),
    }
except ImportError:
    SyntaxHighlighter = None



class ScriptController(SourceEditController):

    def __init__(self, document, model=None):
        if model is None: model = ScriptModel()
        SourceEditController.__init__(self, document, model)

    def create_source_editor(self, parent=None):
        edit = QtGui.QPlainTextEdit(parent)
        edit.setFont(DEFAULT_FONT)
        if hasPyCode:
            self.pyedit = PyEdit(".", edit)
        if SyntaxHighlighter:
            parts_scanner, code_scanner, formats = load_syntax(syntax, scheme)
            self.highlighter = SyntaxHighlighter(edit.document(), parts_scanner, code_scanner, formats, default_font=DEFAULT_FONT)
        edit.setReadOnly(self.model.is_read_only())
        return edit

    def on_edit_enter(self):
        super(ScriptController, self).on_edit_enter()
        if hasPyCode:
            self.pyedit.prefix = self.document.stubs()
