# -*- coding: utf-8 -*-
from collections import namedtuple

from ..qt import Qt
from ..qt.QtCore import SIGNAL, QAbstractListModel, Qt, QModelIndex, QAbstractTableModel, QThread
from ..qt.QtGui import QToolTip, QTreeWidget, QTreeWidgetItem, QTextCursor, QFrame, QIcon, \
    QCompleter, QTableView, QAbstractItemView, QKeyEvent
try:
    from ..qt.QtCore import Signal
except ImportError:
    from ..qt.QtCore import pyqtSignal as Signal


from pycode.ropeassist import completions, calltip, definition_location
from pycode.indenter import PythonCodeIndenter


class BackgroundOperation(QThread):

    def __init__(self, target, args, kwargs):
        super(BackgroundOperation, self).__init__()
        self.target = target
        self.args = args
        self.callback = kwargs.pop("callback", None)
        self.kwargs = kwargs
        self.result = None
        self.finished.connect(self.on_finished)

    def run(self):
        self.result = self.target(*self.args, **self.kwargs)

    def on_finished(self):
        # slot called in main (gui) thread
        if self.callback:
            self.callback(self.result)
        # To be safe, remove refernces for garbage collection
        self.args = self.kwargs = self.target = None


def call_background(func, *args, **kwargs):
    func._thread = BackgroundOperation(func, args=args, kwargs=kwargs)
    func._thread.start()


class RopeEditorAdapter(object):

    """
    Wrapper for ..qt to provide the Line-editor interface
    for the Indenter classes
    """

    def __init__(self, textedit):
        self.textedit = textedit

    def length(self):
        return len(self.textedit.toPlainText())

    def get_line(self, line_no=None):
        block = self.textedit.document().findBlockByNumber(line_no)
        return unicode(block.text())

    def insert(self, pos, s):
        cursor = QTextCursor(self.textedit())
        cursor.setPosition(pos)
        cursor.insertText(s)

    def indent_line(self, line_no, indent_length):
        block = self.textedit.document().findBlockByNumber(line_no)
        cursor = QTextCursor(block)
        cursor.joinPreviousEditBlock()
        cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.MoveAnchor)
        if indent_length < 0:
            for i in range(-indent_length):
                cursor.deleteChar()
        else:
            cursor.insertText(" " * indent_length)
        if indent_length:
            cursor.movePosition(
                QTextCursor.StartOfBlock, QTextCursor.MoveAnchor)
            line = unicode(cursor.block().text())
            if len(line) and line[0] == " ":
                cursor.movePosition(
                    QTextCursor.NextWord, QTextCursor.MoveAnchor)
            self.textedit.setTextCursor(cursor)
        cursor.endEditBlock()


class OutlineItem(QTreeWidgetItem):

    _icon_map = {}

    def __init__(self, parent, node):
        QTreeWidgetItem.__init__(self, parent)
        kind = node.get_kind()
        if not self._icon_map:
            self._load_icons()
        self.setIcon(0, self._icon_map[kind])
        name = node.name
        self.key = "%s.%s" % (parent.key, name)
        self.setText(0, name)
        self.node = node
        self.items = []
        root = self.treeWidget()
        eflag = root.is_expanded.get(self.key, True)
        self.setExpanded(eflag)
        for child in node.get_children():
            i = self.__class__(self, child)
            self.items.append(i)
            self.addChild(i)

    def _load_icons(self):
        self._icon_map.update({
                              "function": QIcon.fromTheme("code-function"),
                              "class": QIcon.fromTheme("code-class")})


class OutlineTree(QTreeWidget):

    def __init__(self, parent=None):
        super(OutlineTree, self).__init__(parent)
        self.setFrameStyle(QFrame.NoFrame)
        self.key = ""
        self.is_expanded = {}
        self.setHeaderLabels(["Name"])
        self.setHeaderHidden(True)
        self.connect(
            self, SIGNAL("itemActivated(QTreeWidgetItem*,int)"), self.on_activated)
        self.connect(
            self, SIGNAL("itemExpanded(QTreeWidgetItem*)"), self.on_expanded)
        self.connect(
            self, SIGNAL("itemCollapsed(QTreeWidgetItem*)"), self.on_collapsed)
        # self.setRootIsDecorated(False)

    def on_expanded(self, item):
        self.tree.is_expanded[item.key] = True

    def on_collapsed(self, item):
        self.tree.is_expanded[item.key] = False

    def refresh(self, source_code):
        # XXX
        self.rope_outline.get_root_nodes(source_code)

    def set_nodes(self, nodes):
        del self.items[:]
        self.clear()
        if not nodes:
            return
        for n in nodes:
            i = OutlineItem(self.tree, n)
            self.items.append(i)
        #self.setColumnWidth(0, 48)
        self.resizeColumnToContents(0)

    def on_activated(self, item, item_col):
        row = item.node.get_line_number() - 1
        self.editor.set_row_start(row)
        self.editor.centerCursor()
        self.editor.setFocus()


class Completion(namedtuple("Completion", "kind name")):
    pass


class CompleterModel(QAbstractTableModel):

    _icon_map = {}

    def __init__(self, items, *columns):
        QAbstractTableModel.__init__(self)
        self.items = items
        self.columns = columns

        if not self._icon_map:
            self._load_icons()

    def _load_icons(self):
        func_icon = QIcon.fromTheme("code-function")
        var_icon = QIcon.fromTheme("code-variable")
        class_icon = QIcon.fromTheme("code-class")
        builtin_icon = QIcon.fromTheme("code-typedef")
        imported_icon = QIcon.fromTheme("code-block")
        instance_icon = class_icon
        module_icon = QIcon.fromTheme("code-context")
        self._icon_map.update({
                              None: QIcon(),
                              "function": func_icon,
                              "variable": var_icon,
                              "parameter": var_icon,
                              "class": class_icon,
                              "builtin": builtin_icon,
                              "imported": imported_icon,
                              "instance": instance_icon,
                              "module": module_icon,
                              })

    # def headerData(self, section, orientation, role=Qt.DisplayRole):
    #    return self.columns[section]
    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role in (Qt.DisplayRole, Qt.EditRole, Qt.DecorationRole):
            row = index.row()
            col = index.column()
            value = self.items[row][col]
            if col == 0:
                value = self._icon_map[value]
            return value
        else:
            return None

    def rowCount(self, parent=QModelIndex()):
        return len(self.items)

    def columnCount(self, parent=QModelIndex()):
        return len(self.columns)


class CompleterPopup(QTableView):

    def __init__(self, textedit):
        super(CompleterPopup, self).__init__(None)
        self.setMinimumHeight(150)
        self._textedit = textedit
        # self.setAlternatingRowColors(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setSortingEnabled(False)
        self.setShowGrid(False)
        self.setWordWrap(False)

    visibility_changed = Signal(bool)

    def showEvent(self, event):
        self.visibility_changed.emit(True)

    def hideEvent(self, event):
        self.visibility_changed.emit(False)


class Completer(QCompleter):

    def __init__(self, textedit, sorting=False, case_sensitivity=False):
        super(Completer, self).__init__(textedit)
        self._textedit = textedit
        self._popup = CompleterPopup(self._textedit)
        self._popup.visibility_changed[
            bool].connect(self._on_visibility_changed)
        self.setPopup(self._popup)
        self.setCompletionColumn(1)
        # self.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self.setWidget(self._textedit)
        self.setCaseSensitivity(Qt.CaseSensitive if case_sensitivity else Qt.CaseInsensitive)
        self.activated[str].connect(self.insert_completion)
        self._textedit.removeEventFilter(self)

    def text_under_cursor(self):
        tc = self._textedit.textCursor()
        tc.select(QTextCursor.WordUnderCursor)
        return tc.selectedText()

    def insert_completion(self, completion):
        tc = self._textedit.textCursor()
        prefix_length = len(unicode(self.text_under_cursor()))
        suffix = unicode(completion)[prefix_length:]
        tc.insertText(suffix)
        self._textedit.setTextCursor(tc)

    def _on_visibility_changed(self, is_visible):
        if is_visible:
            self._textedit.installEventFilter(self)
        else:
            self._textedit.removeEventFilter(self)

    def eventFilter(self, obj, event):
        # TODO: test how many events really arrive
        if isinstance(event, QKeyEvent) and self._on_key_pressed(event):
            return True
        return super(Completer, self).eventFilter(obj, event)

    def _on_key_pressed(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.completionMode() != QCompleter.UnfilteredPopupCompletion:
                sel_list = self._popup.selectionModel().selectedIndexes()
                if sel_list:
                    idx = sel_list[1]
                else:
                    idx = None
            else:
                idx = self._popup.currentIndex()
            if idx:
                activated = self.completionModel().data(
                    idx, Qt.DisplayRole)
                self.insert_completion(activated)
            self._popup.hide()
            return True
        if event.text():
            prefix = self.text_under_cursor()
            self.setCompletionPrefix(prefix)
            self._popup.setCurrentIndex(self.completionModel().index(0, 0))
            return
        elif event.key() in (Qt.Key_Shift, Qt.Key_Control,
                             Qt.Key_Backspace,
                             Qt.Key_Down, Qt.Key_Up, Qt.Key_PageDown, Qt.Key_PageUp):
            return
        else:
            self._popup.hide()
            return True

    def on_completitions_found(self, items):
        """
        callback
        """
        if not items:
            return
        elif len(items) == 1:
            self.insert_completion(items[0].name)
            return

        self._model = CompleterModel(items, "type", "suggestion")
        self.setModel(self._model)
        self._popup.resizeColumnsToContents()
        self._popup.resizeRowsToContents()
        self._popup.updateGeometries()
        #self._popup.sortByColumn(1, Qt.AscendingOrder)
        # self._popup.setModel(self._model)
        #self._popup.setCurrentIndex(self._model.index(0, 0))
        self.setCurrentRow(0)

        cr = self._textedit.cursorRect()
        cr.setWidth(
            self._popup.sizeHintForColumn(0) + self._popup.sizeHintForColumn(1)
            + self._popup.verticalScrollBar().sizeHint().width())

        self._popup.setCurrentIndex(self.completionModel().index(0, 0))
        self.complete(cr)


class CallTip(object):

    def __init__(self, textedit):
        self._textedit = textedit

    def format(self, signature, doc=""):
        call, sep, args = signature.partition("(")
        args, sep, rest = args.rpartition(")")
        alist = []
        for a in args.split(","):
            k, sep, v = a.partition("=")
            s = '<b>%s</b>' % k.strip()
            if v:
                s += "=<i>%s</i>" % v.replace(
                    " ", "&nbsp").replace(
                    "<", "&lt;").replace(
                    ">", "&gt;")
            alist.append(s)
        if alist and alist[0] == "<b>self</b>":
            alist[0] = '<font color="#909090">self</font>'
        args = ",&nbsp;".join(alist)
        t = '<font color="black">%s(%s)%s</font>' % (call, args, rest)
        # if doc:
        #    t += "<br><pre>%s</pre>" % doc
        return t

    def show(self, signature, doc=""):
        cr = self._textedit.cursorRect()
        cr.setX(cr.x() + self._textedit.viewport().x())
        cr.setY(cr.y() + self._textedit.viewport().y())
        pos = self._textedit.mapToGlobal(cr.topLeft())
        text = self.format(signature, doc)
        QToolTip.showText(pos, text, self._textedit)

    def hide(self):
        QToolTip.hideText()

    def isVisible(self):
        return QToolTip.isVisible()


class PyCode(object):

    calltip_factory = CallTip

    def __init__(self, project_folder, textedit, filename=None):
        self._prj = project_folder
        self._textedit = textedit
        self._filename = filename
        self._completer = Completer(self._textedit)
        self._calltip = self.calltip_factory(self._textedit)
        self._indenter = PythonCodeIndenter(RopeEditorAdapter(self._textedit))
        self._textedit_keyPressEvent = self._textedit.keyPressEvent
        self._textedit.keyPressEvent = self.keyPressEvent

    def source(self):
        src = unicode(self._textedit.toPlainText())
        pos = self._textedit.textCursor().position()
        return src, pos

    def request_completion(self):
        src, pos = self.source()
        call_background(completions, self._prj,
                        src, pos, callback=self.got_completions)

    def got_completions(self, result):
        clist = [Completion(c[0], c[1]) for c in result]
        self._completer.on_completitions_found(clist)

    def request_definition(self):
        src, pos = self.source()
        call_background(definition_location, self._prj,
                        src, pos, callback=self.got_definition)

    def got_definition(self, result):
        print "defined at", result

    def request_calltip(self):
        src, pos = self.source()
        call_background(
            calltip, self._prj, src, pos - 1, callback=self.got_calltip)

    def got_calltip(self, result):
        (signature, doc) = result
        if signature:
            self._calltip.show(signature, doc)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        if key == Qt.Key_Tab:
            tc = self._textedit.textCursor()
            lineno = tc.blockNumber()
            self._indenter.correct_indentation(lineno)
            return
        elif key == Qt.Key_Backspace:
            tc = self._textedit.textCursor()
            lineno = tc.blockNumber()
            col = tc.columnNumber()
            text = unicode(tc.block().text())[:col]
            if text and text.strip(" \t") == "":
                self._indenter.deindent(lineno)
                return

        self._textedit_keyPressEvent(event)
        if event.text() and self._calltip.isVisible():
            self._calltip.hide()
        if (key == Qt.Key_Space and modifiers & Qt.ControlModifier) \
           or key == Qt.Key_Period:
            self.request_completion()
        elif key == Qt.Key_Return:
            if modifiers & Qt.ControlModifier:
                self.request_definition()
            else:
                lineno = self._textedit.textCursor().blockNumber()
                self._indenter.entering_new_line(lineno)
        elif key == Qt.Key_ParenLeft:
            self.request_calltip()
