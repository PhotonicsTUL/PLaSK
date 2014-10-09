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
from ..model.materials import MaterialsModel, MaterialPropertyModel, material_html_help, \
    parse_material_components, elements_re
from ..utils.widgets import HTMLDelegate, table_last_col_fill
from . import Controller
from .defines import DefinesCompletionDelegate
from .table import table_and_manipulators, table_with_manipulators

try:
    import plask
except ImportError:
    plask = None
else:
    import plask.material


class ComponentsPopup(QtGui.QFrame):

    def __init__(self, index, name, groups, doping, pos=None):
        super(ComponentsPopup, self).__init__()
        self.setWindowFlags(QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint)
        self.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Plain)
        self.index = index
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
        self.index.model().popup = None
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
        self.index.model().setData(self.index, mat)


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

        combo = QtGui.QComboBox(parent)
        combo.setEditable(True)
        combo.setInsertPolicy(QtGui.QComboBox.NoInsert)
        combo.addItems(material_list)
        combo.setEditText(index.data())
        try: combo.setCurrentIndex(material_list.index(index.data()))
        except ValueError: pass
        combo.insertSeparator(4)
        combo.insertSeparator(len(material_list)-index.row()+1)
        combo.setCompleter(self.get_defines_completer(parent))
        combo.setMaxVisibleItems(len(material_list))
        #self.connect(combo, QtCore.SIGNAL("currentIndexChanged(int)"),
        #             self, QtCore.SLOT("currentIndexChanged()"))
        combo.currentIndexChanged[str].connect(lambda text: self.show_components_popup(combo, text, index))
        return combo

    def show_components_popup(self, combo, text, index):
        pos = combo.mapToGlobal(QtCore.QPoint(0, combo.height()))
        self.popup = None  # close old popup
        name, groups, doping = parse_material_components(text)
        if not groups and doping is None:
            return
        self.popup = ComponentsPopup(index, name, groups, doping, pos)
        self.popup.show()


class MaterialPropertiesDelegate(DefinesCompletionDelegate):

    def __init__(self, defines_model, parent):
        DefinesCompletionDelegate.__init__(self, defines_model, parent)

    def createEditor(self, parent, option, index):
        opts = index.model().options_to_choose(index)

        if index.column() == 0:
            used = [index.model().get(0, i) for i in range(index.model().rowCount()) if i != index.row()]
            opts = [opt for opt in opts if opt not in used]

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


class MaterialsController(Controller):

    def __init__(self, document, selection_model=None):
        if selection_model is None: selection_model = MaterialsModel()
        Controller.__init__(self, document, selection_model)

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

        self.property_model = MaterialPropertyModel(selection_model)
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
        self.splitter.addWidget(table_with_manipulators(self.properties_table, self.splitter,
                                                        title="Properties of the material"))

        self.splitter.setSizes([10000, 30000])

        self.materials_table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.materials_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        selection_model = self.materials_table.selectionModel()
        selection_model.selectionChanged.connect(self.material_selected) #currentChanged ??

        self.properties_table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.properties_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)

    def material_selected(self, new_selection, old_selection):
        indexes = new_selection.indexes()
        if indexes:
            self.property_model.material = self.model.entries[indexes[0].row()]
        else:
            self.property_model.material = None
        #self.properties_table.resizeColumnsToContents()
        self.properties_table.resizeRowsToContents()

    def get_widget(self):
        return self.splitter


