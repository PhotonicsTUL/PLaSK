# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import itertools

from ..qt import QtCore, QtGui

from ..external.highlighter import SyntaxHighlighter, load_syntax
from ..external.highlighter.python27 import syntax
from .script import scheme

from ..model.materials import MaterialsModel, MaterialPropertyModel, material_html_help, \
    parse_material_components, elements_re
from ..utils.textedit import TextEdit
from ..utils.widgets import HTMLDelegate, table_last_col_fill, DEFAULT_FONT, table_edit_shortcut
from ..utils.qsignals import BlockQtSignals
from . import Controller
from .defines import DefinesCompletionDelegate
from .table import table_and_manipulators, table_with_manipulators
from .defines import get_defines_completer

try:
    import plask
except ImportError:
    plask = None
else:
    import plask.material


class ComponentsPopup(QtGui.QFrame):

    def __init__(self, close_cb, name, groups, doping, pos=None):
        super(ComponentsPopup, self).__init__()
        self.setWindowFlags(QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint)
        self.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Plain)
        self.close_cb = close_cb
        self.elements = elements_re.findall(name)
        self.doping = doping
        self.edits = {}
        first = None
        box = QtGui.QHBoxLayout()
        for el in tuple(itertools.chain(*(g for g in groups if len(g) > 1))):
            label = QtGui.QLabel(' ' + el + ':')
            edit = QtGui.QLineEdit(self)
            if first is None: first = edit
            box.addWidget(label)
            box.addWidget(edit)
            self.edits[el] = edit
        if doping:
            label = QtGui.QLabel(' [' + doping + ']:')
            edit = QtGui.QLineEdit(self)
            if first is None: first = edit
            box.addWidget(label)
            box.addWidget(edit)
            self.edits['dp'] = edit
        box.setContentsMargins(2, 1, 2, 1)
        self.setLayout(box)
        if pos is None:
            cursor = QtGui.QCursor()
            self.move(cursor.pos())
        else:
            self.move(pos)
        if first: first.setFocus()

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Escape):
            self.close()

    def closeEvent(self, event):
        #self.index.model().popup = None
        mat = ''
        for el in self.elements:
            mat += el
            if self.edits.has_key(el):
                val = str(self.edits[el].text())
                if val: mat += '(' + val + ')'
        if self.doping:
            mat += ':' + self.doping
            val = str(self.edits['dp'].text())
            if val: mat += '=' + val
        #self.index.model().setData(self.index, mat)
        self.close_cb(mat)


class MaterialsComboBox(QtGui.QComboBox):

    def __init__(self, parent=None, material_list=None, defines_model=None, popup_select_cb=None):
        """
        :param parent: Qt Object parent
        :param material_list: list of materials to add
        :param defines_model: defines model used to completion
        :param popup_select_cb: called after selecting components in ComponentsPopup (it can be called after deleting internal QComboBox)
        """
        super(MaterialsComboBox, self).__init__(parent)
        self.popup_select_cb = popup_select_cb
        self.setEditable(True)
        self.setInsertPolicy(QtGui.QComboBox.NoInsert)
        if defines_model is not None: self.setCompleter(get_defines_completer(defines_model, parent))
        if material_list:
            self.addItems(material_list)
            self.setMaxVisibleItems(len(material_list))
        self.currentIndexChanged[str].connect(self.show_components_popup)

    def append_list(self, list_to_append, insert_separator = True):
        """
        Append list to combo-box.
        :param list list_to_append: list to append
        :param bool insert_separator: if True (default) separator will be added between existing items and appended once
          (separator will be inserted only if both sets are not empty)
        """
        if list_to_append:
            if insert_separator and self.count() > 0: self.insertSeparator(self.count())
            self.addItems(list_to_append)

    def append_materials_from_model(self, material_model, insert_separator = True):
        self.append_list([e.name for e in material_model.entries], insert_separator)

    def append_materials_from_db(self, db = None, insert_separator = True):
        self.append_list(sorted(db if db is not None else plask.material.db, key=lambda x: x.lower()), insert_separator)

    def show_components_popup(self, text):
        pos = self.mapToGlobal(QtCore.QPoint(0, self.height()))
        self.material_edit_popup = None  # close old popup
        name, groups, doping = parse_material_components(text)
        if not groups and doping is None:
            return
        self.material_edit_popup = ComponentsPopup(self.close_popup, name, groups, doping, pos)
        self.material_edit_popup.show()

    def close_popup(self, material_name):
        self.material_edit_popup = None
        try:
            self.setEditText(material_name)
        except:
            pass # it is possible that internal combo box has been deleted
        if self.popup_select_cb is not None: self.popup_select_cb(material_name)


class MaterialBaseDelegate(DefinesCompletionDelegate):

    @staticmethod
    def _format_material(mat):
        return mat

    def __init__(self, defines_model, parent):
        DefinesCompletionDelegate.__init__(self, defines_model, parent)

    def createEditor(self, parent, option, index):

        material_list = ['dielectric', 'liquid_crystal', 'metal', 'semiconductor']

        if plask:
            material_list.extend(
                sorted((self._format_material(mat) for mat in plask.material.db if mat not in material_list),
                       key=lambda x: x.lower()))

        material_list.extend(e.name for e in index.model().entries[0:index.row()])

        if not material_list: return super(MaterialBaseDelegate, self).createEditor(parent, option, index)

        combo = MaterialsComboBox(parent, material_list, self.model, popup_select_cb = lambda mat: index.model().setData(index, mat))
        combo.setEditText(index.data())
        try: combo.setCurrentIndex(material_list.index(index.data()))
        except ValueError: pass
        combo.insertSeparator(4)
        combo.insertSeparator(len(material_list)-index.row()+1)
        combo.setMaxVisibleItems(len(material_list))
        #self.connect(combo, QtCore.SIGNAL("currentIndexChanged(int)"),
        #             self, QtCore.SLOT("currentIndexChanged()"))
        #combo.currentIndexChanged[str].connect()
        return combo


class MaterialPropertiesDelegate(DefinesCompletionDelegate):

    def __init__(self, defines_model, parent):
        DefinesCompletionDelegate.__init__(self, defines_model, parent)

    def createEditor(self, parent, option, index):
        opts = index.model().options_to_choose(index)

        if index.column() == 0:
            used = [index.model().get(0, i) for i in range(index.model().rowCount()) if i != index.row()]
            opts = [opt for opt in opts if opt not in used]
            self._first_enter = True

        if opts is None: return super(MaterialPropertiesDelegate, self).createEditor(parent, option, index)

        combo = QtGui.QComboBox(parent)
        combo.setInsertPolicy(QtGui.QComboBox.NoInsert)
        combo.addItems(opts)
        combo.setMaxVisibleItems(len(opts))
        if index.column() == 0:
            try:
                combo.setCurrentIndex(opts.index(index.data()))
            except ValueError:
                combo.setCurrentIndex(0)
            combo.highlighted.connect(lambda i:
                QtGui.QToolTip.showText(QtGui.QCursor.pos(), material_html_help(combo.itemText(i))))
        else:
            combo.setEditable(True)
            combo.setEditText(index.data())
            completer = combo.completer()
            completer.setCaseSensitivity(QtCore.Qt.CaseSensitive)
            combo.setCompleter(completer)
        #combo.setCompleter(completer)
        #self.connect(combo, QtCore.SIGNAL("currentIndexChanged(int)"),
        #             self, QtCore.SLOT("currentIndexChanged()"))

        return combo

    def eventFilter(self, editor, event):
        if isinstance(editor, QtGui.QComboBox) and event.type() == QtCore.QEvent.Enter and self._first_enter:
            editor.showPopup()
            self._first_enter = False
            return True
        else:
            return super(MaterialPropertiesDelegate, self).eventFilter(editor, event)


class MaterialsController(Controller):

    def __init__(self, document, material_selection_model=None):
        if material_selection_model is None: material_selection_model = MaterialsModel()
        Controller.__init__(self, document, material_selection_model)

        self.splitter = QtGui.QSplitter()

        self.materials_table = QtGui.QTableView()
        self.materials_table.setModel(self.model)
        self.materials_table.setItemDelegateForColumn(1, MaterialBaseDelegate(self.document.defines.model,
                                                                              self.materials_table))
        #self.materialsTableActions = TableActions(self.materials_table)
        table_last_col_fill(self.materials_table, self.model.columnCount(None), 140)
        materials_table, materials_toolbar = \
            table_and_manipulators(self.materials_table, self.splitter, title="Materials")
        self.splitter.addWidget(materials_table)
        # materials_toolbar.addSeparator()
        # materials_toolbar.addAction(self.document.window.material_plot_action)

        prop_splitter = QtGui.QSplitter()
        prop_splitter.setOrientation(QtCore.Qt.Vertical)

        self.property_model = MaterialPropertyModel(material_selection_model)
        self.properties_table = QtGui.QTableView()
        self.properties_table.setModel(self.property_model)
        self.properties_delegate = MaterialPropertiesDelegate(self.document.defines.model, self.properties_table)
        self.unit_delegate = HTMLDelegate(self.properties_table)
        self.help_delegate = HTMLDelegate(self.properties_table)
        self.properties_table.setItemDelegateForColumn(0, self.properties_delegate)
        self.properties_table.setItemDelegateForColumn(1, self.properties_delegate)
        self.properties_table.setItemDelegateForColumn(2, self.unit_delegate)
        self.properties_table.setItemDelegateForColumn(3, self.help_delegate)
        #self.properties_table.setWordWrap(True)
        table_last_col_fill(self.properties_table, self.property_model.columnCount(None), [90, 180, 50])

        self.properties_table.verticalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        prop_splitter.addWidget(table_with_manipulators(self.properties_table, self.splitter,
                                                        title="Properties of the material"))

        self.materials_table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.materials_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.materials_table.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        table_edit_shortcut(self.materials_table, 0, 'n')
        table_edit_shortcut(self.materials_table, 1, 'b')
        material_selection_model = self.materials_table.selectionModel()
        material_selection_model.selectionChanged.connect(self.material_selected)

        self.properties_table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.properties_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.properties_table.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        table_edit_shortcut(self.properties_table, 0, 'n')
        table_edit_shortcut(self.properties_table, 1, 'v')
        property_selection_model = self.properties_table.selectionModel()
        property_selection_model.selectionChanged.connect(self.property_selected)

        font = QtGui.QFont(DEFAULT_FONT)
        # font.setPointSize(font.pointSize()-1)
        self.propedit = TextEdit(prop_splitter, line_numbers=False)
        self.propedit.highlighter = SyntaxHighlighter(self.propedit.document(), *load_syntax(syntax, scheme),
                                                      default_font=font)
        self.propedit.hide()

        self.property_model.dataChanged.connect(self.property_data_changed)

        prop_splitter.addWidget(self.propedit)
        prop_splitter.setSizes([50000, 10000])
        self.splitter.addWidget(prop_splitter)
        self.splitter.setSizes([10000, 30000])

    def material_selected(self, new_selection, old_selection):
        self.propedit.hide()
        indexes = new_selection.indexes()
        if indexes:
            self.property_model.material = self.model.entries[indexes[0].row()]
        else:
            self.property_model.material = None
        #self.properties_table.resizeColumnsToContents()
        self.properties_table.resizeRowsToContents()

    def property_selected(self, new_selection, old_selection):
        indexes = new_selection.indexes()
        if indexes:
            try: self.propedit.textChanged.disconnect()
            except RuntimeError: pass
            row = indexes[0].row()
            self.propedit.setPlainText(self.property_model.material.properties[row][1])
            self.propedit.show()
            self.propedit.textChanged.connect(lambda: self.propedit_changed(row))
        else:
            try: self.propedit.textChanged.disconnect()
            except RuntimeError: pass
            self.propedit.hide()

    def propedit_changed(self, row):
        self.property_model.dataChanged.disconnect(self.property_data_changed)
        try:
            self.property_model.setData(self.property_model.createIndex(row, 1), self.propedit.toPlainText())
        finally:
            self.property_model.dataChanged.connect(self.property_data_changed)

    def property_data_changed(self, tl, br):
        with BlockQtSignals(self.propedit):
            self.propedit.setPlainText(self.property_model.get(1, tl.row()))

    def get_widget(self):
        return self.splitter


