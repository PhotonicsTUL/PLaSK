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

from ...qt.QtCore import *
from ...qt.QtWidgets import *
from ...qt.QtGui import *
from ...qt import QtSignal
from ...external.highlighter import SyntaxHighlighter, load_syntax
from ...external.highlighter.python27 import syntax
from ..script import scheme
from ...model.materials import MaterialsModel, material_html_help, parse_material_components, elements_re
from ...utils.texteditor import TextEditor
from ...utils.widgets import HTMLDelegate, table_last_col_fill, EDITOR_FONT, table_edit_shortcut, CheckBoxDelegate
from ...utils.qsignals import BlockQtSignals
from .. import Controller, select_index_from_info
from ..defines import DefinesCompletionDelegate
from ..table import table_and_manipulators, table_with_manipulators, TableActions
from ..defines import get_defines_completer
from .plot import show_material_plot


try:
    import plask
except ImportError:
    plask = None
else:
    import plask.material


BASE_MATERIALS = ['dielectric', 'liquid_crystal', 'metal', 'semiconductor']


class ComponentsPopup(QFrame):

    def __init__(self, close_cb, name, label, groups, doping, pos=None):
        super(ComponentsPopup, self).__init__()
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.close_cb = close_cb
        self.elements = elements_re.findall(name)
        self.label = label
        self.doping = doping
        self.edits = {}
        first = None
        box = QHBoxLayout()
        for el in tuple(itertools.chain(*((e[0] for e in g) for g in groups if len(g) > 1))):
            qlabel = QLabel(' ' + el + ':')
            edit = QLineEdit(self)
            if first is None: first = edit
            box.addWidget(qlabel)
            box.addWidget(edit)
            self.edits[el] = edit
        if doping:
            qlabel = QLabel(' [' + doping + ']:')
            edit = QLineEdit(self)
            if first is None: first = edit
            box.addWidget(qlabel)
            box.addWidget(edit)
            self.edits['dp'] = edit
        box.setContentsMargins(2, 1, 2, 1)
        self.setLayout(box)
        if pos is None:
            cursor = QCursor()
            self.move(cursor.pos())
        else:
            self.move(pos)
        if first: first.setFocus()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
            self.close()

    def closeEvent(self, event):
        #self.index.model().popup = None
        mat = ''
        for el, _ in self.elements:
            mat += el
            if el in self.edits:
                val = str(self.edits[el].text())
                if val: mat += '(' + val + ')'
        if self.label:
            mat += '_'
            mat += self.label
        if self.doping:
            mat += ':' + self.doping
            val = str(self.edits['dp'].text())
            if val: mat += '=' + val
        #self.index.model().setData(self.index, mat)
        self.close_cb(mat)


class MaterialLineEdit(QLineEdit):

    def __init__(self, parent, materials_model):
        super(MaterialLineEdit, self).__init__(parent)

        self.materials_model = materials_model

        # Create a clear button with icon
        self.button = QToolButton(self)
        self.button.setIcon(QIcon.fromTheme('matplotlib'))
        self.button.setCursor(Qt.ArrowCursor)
        self.button.setStyleSheet("QToolButton { border: none; padding: 0px; }")

        # signals, clear lineEdit if btn pressed; change btn visibility on input
        self.button.clicked.connect(self.show_material)

        frw = self.style().pixelMetric(QStyle.PM_DefaultFrameWidth)
        self.setStyleSheet("QLineEdit {{ padding-right: {}px; }} ".format(self.button.sizeHint().width() + frw + 1))
        msh = self.minimumSizeHint().height()
        self.button.setMaximumHeight(msh)

    def resizeEvent(self, event):
        sz = self.button.sizeHint()
        frw = self.style().pixelMetric(QStyle.PM_DefaultFrameWidth)
        self.button.move(self.rect().right() - frw - sz.width(), (self.rect().bottom() + 1 - sz.height()) / 2)

    def show_material(self):
        show_material_plot(self, self.materials_model, self.text())


class MaterialsComboBox(QComboBox):

    editingFinished = QtSignal()

    def __init__(self, parent=None, materials_model=None, material_list=None, defines_model=None, popup_select_cb=None):
        """
        :param parent: Qt Object parent
        :param material_list: list of materials to add
        :param defines_model: defines model used to completion
        :param popup_select_cb: called after selecting components in ComponentsPopup
               (it can be called after deleting internal QComboBox)
        """
        super(MaterialsComboBox, self).__init__(parent)
        if materials_model is not None:
            line_edit = MaterialLineEdit(self, materials_model)
            self.setLineEdit(line_edit)
        self.popup_select_cb = popup_select_cb
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        if defines_model is not None: self.setCompleter(get_defines_completer(defines_model, parent, material_list))
        if material_list:
            self.addItems(material_list)
            self.setMaxVisibleItems(len(material_list))
        self.currentIndexChanged[str].connect(self.show_components_popup)
        self.material_edit_popup = None

    def focusOutEvent(self, event):
        if self.material_edit_popup is None:
            self.editingFinished.emit()
        super(MaterialsComboBox, self).focusOutEvent(event)

    def keyPressEvent(self, event):
        super(MaterialsComboBox, self).keyPressEvent(event)
        if event.key() in (Qt.Key_Enter, Qt.Key_Return):
            self.editingFinished.emit()

    def append_list(self, list_to_append, insert_separator=True):
        """
        Append list to combo-box.
        :param list list_to_append: list to append
        :param bool insert_separator: if True (default) separator will be added between existing items and appended once
          (separator will be inserted only if both sets are not empty)
        """
        if list_to_append:
            if insert_separator and self.count() > 0: self.insertSeparator(self.count())
            self.addItems(list_to_append)

    def append_materials_from_model(self, material_model, insert_separator=True):
        self.append_list([e.name for e in material_model.entries], insert_separator)

    def append_materials_from_db(self, db=None, insert_separator=True):
        self.append_list([m for m in sorted(db if db is not None else plask.material.db, key=lambda x: x.lower())
                          if m not in BASE_MATERIALS], True)
        self.append_list(BASE_MATERIALS, insert_separator)

    def show_components_popup(self, text):
        pos = self.mapToGlobal(QPoint(0, self.height()))
        self.material_edit_popup = None  # close old popup
        name, label, groups, doping = parse_material_components(text)
        if not groups and doping is None:
            return
        self.material_edit_popup = ComponentsPopup(self.close_popup, name, label, groups, doping, pos)
        self.material_edit_popup.show()

    def close_popup(self, material_name):
        self.material_edit_popup = None
        try:
            self.setEditText(material_name)
        except:
            pass # it is possible that internal combo box has been deleted
        if self.popup_select_cb is not None: self.popup_select_cb(material_name)


class MaterialNameDelegate(QStyledItemDelegate):

    def __init__(self, model, parent):
        super(MaterialNameDelegate, self).__init__(parent)
        self.model = model

    def createEditor(self, parent, option, index):
        if not isinstance(self.model.entries[index.row()], MaterialsModel.External):
            ed = MaterialLineEdit(parent, self.model)
        else:
            ed = QLineEdit(parent)
        return ed

    def sizeHint(self, item, index):
        hint = super(MaterialNameDelegate, self).sizeHint(item, index)
        if not isinstance(self.model.entries[index.row()], MaterialsModel.External):
            hint.setWidth(hint.width() + 24)
        return hint


class MaterialBaseDelegate(DefinesCompletionDelegate):

    @staticmethod
    def _format_material(mat):
        return mat

    def __init__(self, defines_model, materials_model, parent):
        DefinesCompletionDelegate.__init__(self, defines_model, parent)
        self.materials_model = materials_model

    def createEditor(self, parent, option, index):

        material_list = BASE_MATERIALS[:]

        if plask:
            material_list.extend(
                sorted((self._format_material(mat) for mat in plask.material.db if mat not in material_list),
                       key=lambda x: x.lower()))

        material_list.extend(e.name for e in index.model().entries[0:index.row()])

        if not material_list: return super(MaterialBaseDelegate, self).createEditor(parent, option, index)

        combo = MaterialsComboBox(parent, self.materials_model, material_list, self.model,
                                  popup_select_cb=lambda mat: index.model().setData(index, mat))
        combo.setEditText(index.data())
        try: combo.setCurrentIndex(material_list.index(index.data()))
        except ValueError: pass
        combo.insertSeparator(4)
        combo.insertSeparator(len(material_list)-index.row()+1)
        combo.setMaxVisibleItems(len(material_list))
        #self.connect(combo, SIGNAL("currentIndexChanged(int)"),
        #             self, SLOT("currentIndexChanged()"))
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

        combo = QComboBox(parent)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.addItems(opts)
        combo.setMaxVisibleItems(len(opts))
        if index.column() == 0:
            try:
                combo.setCurrentIndex(opts.index(index.data()))
            except ValueError:
                combo.setCurrentIndex(0)
            combo.highlighted.connect(lambda i:
                QToolTip.showText(QCursor.pos(), material_html_help(combo.itemText(i))))
        else:
            combo.setEditable(True)
            combo.setEditText(index.data())
            completer = combo.completer()
            completer.setCaseSensitivity(Qt.CaseSensitive)
            combo.setCompleter(completer)
        #combo.setCompleter(completer)
        #self.connect(combo, SIGNAL("currentIndexChanged(int)"),
        #             self, SLOT("currentIndexChanged()"))

        return combo

    def eventFilter(self, editor, event):
        if isinstance(editor, QComboBox) and event.type() == QEvent.Enter and self._first_enter:
            editor.showPopup()
            self._first_enter = False
            return True
        else:
            return super(MaterialPropertiesDelegate, self).eventFilter(editor, event)


class MaterialsController(Controller):

    def __init__(self, document, material_selection_model=None):
        if material_selection_model is None: material_selection_model = MaterialsModel()
        Controller.__init__(self, document, material_selection_model)

        self.selected_material = None

        self.splitter = QSplitter()

        self.materials_table = QTableView()
        self.materials_table.setModel(self.model)
        #self.model.modelReset.connect(lambda : self.materials_table.clearSelection())  #TODO why does not work?
        self.materials_table.setItemDelegateForColumn(0, MaterialNameDelegate(self.model,
                                                                              self.materials_table))
        self.materials_table.setItemDelegateForColumn(1, MaterialBaseDelegate(self.document.defines.model,
                                                                              self.model,
                                                                              self.materials_table))
        self.materials_table.setItemDelegateForColumn(2, CheckBoxDelegate(self.materials_table))


        materials_table, materials_toolbar = \
            table_and_manipulators(self.materials_table, self.splitter, title="Materials")
        library_action = TableActions.make_action('material-library', 'Add &Library',
                                                  'Add new binary library to the list', self.materials_table,
                                                  lambda: self.add_external('library'))
        module_action = TableActions.make_action('material-module', 'Add &Module',
                                                 'Add new python module to the list', self.materials_table,
                                                 lambda: self.add_external('module'))
        tool_button = QToolButton()
        tool_button.setIcon(QIcon.fromTheme('material-external'))
        tool_menu = QMenu(self.materials_table)
        tool_menu.addAction(library_action)
        tool_menu.addAction(module_action)
        tool_button.setMenu(tool_menu)
        tool_button.setPopupMode(QToolButton.InstantPopup)
        materials_toolbar.addWidget(tool_button)
        self.splitter.addWidget(materials_table)
        # materials_toolbar.addSeparator()
        # materials_toolbar.addAction(self.document.window.material_plot_action)

        self.prop_splitter = QSplitter()
        self.prop_splitter.setOrientation(Qt.Vertical)

        #self.property_model = MaterialPropertyModel(material_selection_model)
        self.properties_table = QTableView()
        #self.properties_table.setModel(self.property_model)
        self.properties_delegate = MaterialPropertiesDelegate(self.document.defines.model, self.properties_table)
        self.unit_delegate = HTMLDelegate(self.properties_table)
        self.help_delegate = HTMLDelegate(self.properties_table)
        self.properties_table.setItemDelegateForColumn(0, self.properties_delegate)
        self.properties_table.setItemDelegateForColumn(1, self.properties_delegate)
        self.properties_table.setItemDelegateForColumn(2, self.unit_delegate)
        self.properties_table.setItemDelegateForColumn(3, self.help_delegate)
        #self.properties_table.setWordWrap(True)
        self.prop_splitter.addWidget(table_with_manipulators(self.properties_table, self.splitter,
                                                             title="Properties of the material"))

        self.materials_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.materials_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.materials_table.setColumnWidth(0, 140)
        try:
            self.materials_table.horizontalHeader().setResizeMode(1, QHeaderView.Stretch)
        except AttributeError:
            self.materials_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        check_box_style_option = QStyleOptionButton()
        check_box_rect = QApplication.style().subElementRect(QStyle.SE_CheckBoxIndicator,
                                                                   check_box_style_option, None)
        self.materials_table.setColumnWidth(2, check_box_rect.width()+8)

        table_edit_shortcut(self.materials_table, 0, 'n')
        table_edit_shortcut(self.materials_table, 1, 'b')
        material_selection_model = self.materials_table.selectionModel()
        material_selection_model.selectionChanged.connect(self.material_selected)

        self.model.infoChanged.connect(self.update_materials_table)

        self.properties_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.properties_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table_last_col_fill(self.properties_table, 4, [80, 320, 50])
        table_edit_shortcut(self.properties_table, 0, 'n')
        table_edit_shortcut(self.properties_table, 1, 'v')

        # font.setPointSize(font.pointSize()-1)
        self.propedit = TextEditor(self.prop_splitter, line_numbers=False)
        self.propedit.highlighter = SyntaxHighlighter(self.propedit.document(), *load_syntax(syntax, scheme),
                                                      default_font=EDITOR_FONT)
        self.propedit.hide()

        self.document.window.config_changed.connect(self.reconfig)

        focus_action = QAction(self.materials_table)
        focus_action.triggered.connect(lambda: self.properties_table.setFocus())
        focus_action.setShortcut(QKeySequence(Qt.Key_Return))
        focus_action.setShortcutContext(Qt.WidgetShortcut)
        self.materials_table.addAction(focus_action)

        self.prop_splitter.addWidget(self.propedit)
        self.prop_splitter.setSizes([50000, 10000])
        self.prop_splitter.setEnabled(False)
        self.splitter.addWidget(self.prop_splitter)
        self.splitter.setSizes([10000, 30000])

    def reconfig(self):
        with BlockQtSignals(self.propedit):
            del self.propedit.highlighter
            self.propedit.highlighter = SyntaxHighlighter(self.propedit.document(), *load_syntax(syntax, scheme),
                                                          default_font=EDITOR_FONT)

    def update_materials_table(self, model):
        if model == self.model and model.rowCount():
            index0 = model.index(0, 0)
            index1 = model.index(model.rowCount()-1, model.columnCount()-1)
            model.dataChanged.emit(index0, index1)

    def add_external(self, what):
        index = self.materials_table.selectionModel().currentIndex()
        if index.isValid():
            row = self.model.insert(index.row()+1, value=MaterialsModel.External(self.model, what))
        else:
            row = self.model.insert(value=MaterialsModel.External(self.model, what))
        if row is not None: self.materials_table.selectRow(row)

    def material_selected(self, new_selection, old_selection=None):
        self.propedit.hide()
        indexes = new_selection.indexes()
        if self.selected_material is not None:
            self.selected_material.dataChanged.disconnect(self.property_data_changed)
        if indexes and isinstance(self.model.entries[indexes[0].row()], MaterialsModel.Material):
            #self.property_model.material = self.model.entries[indexes[0].row()]
            self.selected_material = self.model.entries[indexes[0].row()]
            self.properties_table.setModel(self.selected_material)
            property_selection_model = self.properties_table.selectionModel()
            property_selection_model.selectionChanged.connect(self.property_selected)
            self.prop_splitter.setEnabled(True)
            # self.properties_table.resizeColumnsToContents()
            try:
                self.properties_table.horizontalHeader().setResizeMode(2, QHeaderView.ResizeToContents)
            except AttributeError:
                self.properties_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.properties_table.resizeColumnToContents(1)
            self.properties_table.resizeRowsToContents()
        else:
            #self.property_model.material = None
            self.selected_material = None
            self.properties_table.setModel(self.selected_material)
            self.prop_splitter.setEnabled(False)
        if self.selected_material is not None:
            self.selected_material.dataChanged.connect(self.property_data_changed)

    def property_selected(self, new_selection, old_selection):
        indexes = new_selection.indexes()
        try: self.propedit.textChanged.disconnect()
        except (RuntimeError, TypeError): pass
        if indexes:
            row = indexes[0].row()
            self.propedit.setPlainText(self.selected_material.properties[row][1])
            self.propedit.show()
            self.propedit.textChanged.connect(lambda: self.propedit_changed(row))
        else:
            self.propedit.hide()

    def propedit_changed(self, row):
        self.selected_material.dataChanged.disconnect(self.property_data_changed)
        try:
            self.selected_material.setData(self.selected_material.createIndex(row, 1), self.propedit.toPlainText())
        finally:
            self.selected_material.dataChanged.connect(self.property_data_changed)

    def property_data_changed(self, top_left, bottom_right):
        if top_left.row() in (i.row() for i in self.properties_table.selectedIndexes()):
            with BlockQtSignals(self.propedit):
                self.propedit.setPlainText(self.selected_material.get(1, top_left.row()))
        self.properties_table.resizeRowsToContents()  # ensure all documentation is visible

    def get_widget(self):
        return self.splitter

    def on_edit_enter(self):
        try:
            row = self.model.entries.index(self.selected_material)
            self.materials_table.selectRow(row)
        except (ValueError, AttributeError):
            self.materials_table.selectRow(0)
        self.materials_table.setFocus()

    def select_info(self, info):
        if select_index_from_info(info, self.model, self.materials_table):
            #TODO try to select property
            pass
