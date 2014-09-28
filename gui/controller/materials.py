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

from ..qt import QtCore, QtGui, qt

from ..utils.str import html_to_tex
from ..model.materials import MaterialsModel, MaterialPropertyModel, material_html_help, \
                              MATERIALS_PROPERTES, parse_material_components, elements_re
from ..utils.widgets import HTMLDelegate, table_last_col_fill
from .base import Controller
from .defines import DefinesCompletionDelegate
from .table import table_with_manipulators

try:
    import matplotlib
except ImportError:
    matplotlib = None
else:
    import numpy
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

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
        combo.currentIndexChanged[str].connect(lambda text: self.show_components_popup(text, index))
        return combo

    def show_components_popup(self, text, index):
        combo = self.sender()
        pos = combo.mapToGlobal(QtCore.QPoint(0, combo.height()))
        index.model().popup = None  # close old popup
        name, groups, doping = parse_material_components(text)
        if not groups and doping is None:
            return
        index.model().popup = ComponentsPopup(index, name, groups, doping, pos)
        index.model().popup.show()



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
        self.splitter.addWidget(table_with_manipulators(self.materials_table, self.splitter, title="Materials"))

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

        self.splitter.setSizes([10000,30000])

        self.materials_table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.materials_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        selection_model = self.materials_table.selectionModel()
        selection_model.selectionChanged.connect(self.material_selected) #currentChanged ??

        self.properties_table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.properties_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)

        self.plot_action = QtGui.QAction(QtGui.QIcon.fromTheme('edit-find', QtGui.QIcon(':/edit-find.png')),
                                         '&Show plot', self.document.window)
        self.plot_action.setStatusTip('Show parameter plot')
        self.plot_action.triggered.connect(self.show_plot)

    def material_selected(self, new_selection, old_selection):
        indexes = new_selection.indexes()
        if indexes:
            self.property_model.material = self.model.entries[indexes[0].row()]
        else:
            self.property_model.material = None
        #self.properties_table.resizeColumnsToContents()
        self.properties_table.resizeRowsToContents()

    def get_editor(self):
        return self.splitter

    def on_edit_enter(self):
        super(MaterialsController, self).on_edit_enter()
        if matplotlib and plask:
            self.document.window.set_section_actions(self.plot_action)

    def show_plot(self):
        # plot_window = QtGui.QDockWidget("Parameter Plot", self.document.window)
        # plot_window.setFeatures(QtGui.QDockWidget.AllDockWidgetFeatures)
        # plot_window.setFloating(True)
        # plot_window.setWidget(MaterialPlot())
        # self.document.window.addDockWidget(QtCore.Qt.BottomDockWidgetArea, plot_window)
        plot_window = QtGui.QMainWindow(self.document.window)
        plot_window.setWindowTitle("Parameter Plot")
        plot_window.setCentralWidget(MaterialPlot(self.model))
        plot_window.show()


class MaterialPlot(QtGui.QWidget):

    def __init__(self, model, parent=None):
        super(MaterialPlot, self).__init__(parent)
        #self.setContentsMargins(0, 0, 0, 0)

        self.model = model

        self.material = QtGui.QComboBox()
        self.material.setEditable(False)
        self.material.setInsertPolicy(QtGui.QComboBox.NoInsert)
        self.material.setMinimumWidth(180)
        self.material.currentIndexChanged.connect(self.material_changed)
        self.param = QtGui.QComboBox()
        self.param.addItems([k for k in MATERIALS_PROPERTES.keys() if k != 'condtype'])
        self.param.currentIndexChanged.connect(self.property_changed)
        self.param.setMaxVisibleItems(len(MATERIALS_PROPERTES))
        self.param.highlighted.connect(lambda i:
            QtGui.QToolTip.showText(QtGui.QCursor.pos(), material_html_help(self.param.itemText(i)))
        )
        toolbar1 = QtGui.QToolBar()
        toolbar1.addWidget(QtGui.QLabel("Material: "))
        toolbar1.addWidget(self.material)
        toolbar1.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        toolbar2 = QtGui.QToolBar()
        toolbar2.addWidget(QtGui.QLabel("Parameter: "))
        toolbar2.addWidget(self.param)
        toolbar2.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        self.par_toolbar = QtGui.QToolBar()
        self.mat_toolbar = QtGui.QToolBar()
        self.arguments = {}

        self.model.changed.connect(self.update_materials)

        plot = QtGui.QPushButton()
        plot.setText("&Plot")
        plot.pressed.connect(self.update_plot)
        plot.setDefault(True)
        plot.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.figure.set_facecolor(self.palette().color(QtGui.QPalette.Background).name())
        self.canvas.updateGeometry()

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.error = QtGui.QTextEdit(self)
        self.error.setVisible(False)
        self.error.setReadOnly(True)
        self.error.setContentsMargins(0, 0, 0, 0)
        self.error.setFrameStyle(0)
        pal = self.error.palette()
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#ffc"))
        self.error.setPalette(pal)
        self.error.acceptRichText()
        self.error.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)

        hbox1 = QtGui.QHBoxLayout()
        hbox2 = QtGui.QHBoxLayout()
        hbox1.addWidget(toolbar1)
        hbox1.addWidget(self.mat_toolbar)
        hbox2.addWidget(toolbar2)
        hbox2.addWidget(self.par_toolbar)

        hbox2.addWidget(plot)

        layout = QtGui.QVBoxLayout()
        layout.addLayout(hbox1)
        layout.addLayout(hbox2)

        plotbox = QtGui.QVBoxLayout()
        plotbox.addWidget(self.error)
        plotbox.addWidget(self.toolbar)
        plotbox.addWidget(self.canvas)

        splitter = QtGui.QSplitter(self)
        splitter.setOrientation(QtCore.Qt.Vertical)
        plotbox_widget = QtGui.QWidget()
        plotbox_widget.setLayout(plotbox)
        splitter.addWidget(plotbox_widget)

        self.info = QtGui.QTextEdit(self)
        self.info.setAcceptRichText(True)
        self.info.setReadOnly(True)
        self.info.setContentsMargins(0, 0, 0, 0)
        self.info.setFrameStyle(0)
        self.info.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Maximum)

        splitter.addWidget(self.info)

        layout.addWidget(splitter)
        self.setLayout(layout)

        self.update_materials()
        self.property_changed()

    def resizeEvent(self, event):
        if self.error.isVisible():
            self.error.setFixedHeight(self.error.document().size().height())
        if self.info.isVisible():
            self.info.setMaximumHeight(self.info.document().size().height())
        self.figure.set_tight_layout(0)

    def update_materials(self, *args, **kwargs):
        text = self.material.currentText()
        material_list = [e.name for e in self.model.entries]
        sep = len(material_list)
        material_blacklist = ['dielectric', 'liquid_crystal', 'metal', 'semiconductor', 'air']
        if plask:
            material_list.extend(sorted((mat for mat in plask.material.db
                                         if mat not in material_list and mat not in material_blacklist),
                                        key=lambda x: x.lower()))
        self.material.clear()
        self.material.addItems(material_list)
        self.material.insertSeparator(sep)
        if args:
            self.material.setEditText(text)
        self.material.setMaxVisibleItems(len(material_list)-1)

    def set_toolbar(self, toolbar, values, old, what):
        """
        :param what: 0: comonent, 1: doping, 2: property argument
        """
        first = None
        for name,descr in values:
            if name is None:
                toolbar.addSeparator()
                continue
            select = QtGui.QRadioButton()
            select.toggled.connect(self.selected_argument)
            select.setAutoExclusive(False)
            if first is None: first = select
            toolbar.addWidget(select)
            select.setText("{}:".format(name))
            select.descr = descr
            val1 = QtGui.QLineEdit()
            val1.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Fixed)
            toolbar.addWidget(val1)
            sep = toolbar.addWidget(QtGui.QLabel("-"))
            sep.setVisible(False)
            val2 = QtGui.QLineEdit()
            val2.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Fixed)
            if select.text() in old:
                val1.setText(old[select.text()][0])
                val2.setText(old[select.text()][1])
            act2 = toolbar.addWidget(val2)
            act2.setVisible(False)
            self.arguments[select] = val1, val2, (sep, act2), what
        for arg in self.arguments:
            if arg.isChecked(): return
        if first is not None: first.setChecked(True)

    def update_info(self):
        """Update info area"""
        material_name = str(self.material.currentText())
        property_name = str(self.param.currentText())
        # TODO add browsing model if info can be included in XML
        try:
            info = plask.material.db.info(material_name)[property_name]
        except (ValueError, KeyError):
            info = None
        self.info.clear()
        if info:
            text = "<table>"
            for key, value in info.items():
                if type(value) == dict:
                    value = ", ".join("<i>{}</i>: {}".format(*i) for i in value.items())
                elif type(value) == list or type(value) == tuple:
                    value = ", ".join(value)
                text += "<tr><td><b>{}: </b></td><td>{}</td></tr>".format(key, value)
            self.info.setText(text + "</table>")
            self.info.resize(self.info.document().idealWidth(), self.info.document().size().height())
            self.info.show()
            self.info.setMaximumHeight(self.info.document().size().height())
        else:
            self.info.hide()

    def material_changed(self):
        old = dict((k.text(), (v[0].text(), v[1].text())) for k,v in self.arguments.items())
        for child in self.mat_toolbar.children():
            if child in self.arguments:
                del self.arguments[child]
        self.mat_toolbar.clear()
        material = self.material.currentText()

        name, groups, dope = parse_material_components(material)

        if groups:
            elements = tuple(itertools.chain(*(g + [None] for g in groups if len(g) > 1)))[:-1]
            self.set_toolbar(self.mat_toolbar, ((e, "{} fraction".format(e)) for e in elements), old, 0)
        else:
            elements = None

        if dope is not None:
            if elements:
                self.mat_toolbar.addSeparator()
            self.set_toolbar(self.mat_toolbar,
                             [("["+dope+"]", dope + "doping concentration [1/cm<sup>3</sup>]")], old, 1)

        self.update_info()

    def property_changed(self):
        old = dict((k.text(), (v[0].text(), v[1].text())) for k,v in self.arguments.items())
        for child in self.par_toolbar.children():
            if child in self.arguments:
                del self.arguments[child]
        self.par_toolbar.clear()
        self.set_toolbar(self.par_toolbar, MATERIALS_PROPERTES[self.param.currentText()][2], old, 2)

        self.update_info()

    def selected_argument(self):
        button = self.sender()
        checked = button.isChecked()
        for act in self.arguments[button][2]:
            act.setVisible(checked)
        if checked:
            self.arg_button = button
            for other in self.arguments:
                if other != button:
                    other.setChecked(False)

    def _parse_other_args(self, button, cat):
        for k,v in self.arguments.items():
            if k is button or v[3] != cat:
                continue
            val = v[0].text()
            if val:
                try: val = float(val)
                except ValueError: pass
                yield str(k.text())[:-1], val

    def update_plot(self):
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        param = str(self.param.currentText())

        import warnings
        old_showwarning = warnings.showwarning
        warns = []

        def showwarning(message, category, filename, lineno, file=None, line=None):
            message = unicode(message)
            if message not in warns:
                warns.append(message)
        warnings.showwarning = showwarning

        try:
            try:
                start, end = (float(v.text()) for v in self.arguments[self.arg_button][:2])
            except ValueError:
                raise ValueError("Wrong ranges '{}' - '{}'"
                                 .format(*(v.text() for v in self.arguments[self.arg_button][:2])))
            plot_range = plask.linspace(start, end, 1001)
            plot_cat = self.arguments[self.arg_button][3]
            other_args = dict(self._parse_other_args(self.arg_button, 2))
            other_elements = dict(self._parse_other_args(self.arg_button, 0))
            other_elements.update(dict(('dc', v) for k,v in self._parse_other_args(self.arg_button, 1)))
            arg_name = 'dc' if plot_cat == 1 else str(self.arg_button.text())[:-1]
            material_name = str(self.material.currentText())
            while True:  # loop for looking-up the base
                material = [e for e in self.model.entries if e.name == material_name]
                if material:
                    material = material[0]
                    mprop = [p for p in material.properties if p[0] == param]
                    if mprop:
                        expr = mprop[0][1]
                        code = compile(expr, '', 'eval')
                        vals = [eval(code, numpy.__dict__, dict(((arg_name, a),), **other_args))
                                for a in plot_range]
                        break
                    else:
                        material_name = material.base  # and we repeat the loop
                        material = None
                else:
                    break
            if not material:
                if plot_cat == 2:
                    material = plask.material.db.get(material_name, **other_elements)
                    vals = [material.__getattribute__(param)(**dict(((arg_name, a),), **other_args))
                            for a in plot_range]
                else:
                    vals = [plask.material.db.get(material_name, **dict(((arg_name, a),), **other_elements)).
                            __getattribute__(param)(**other_args) for a in plot_range]
            axes.plot(plot_range, vals)
        except Exception as err:
            self.error.setText('<div style="color:red;">{}</div>'.format(str(err)))
            self.error.show()
            self.error.setFixedHeight(self.error.document().size().height())
        else:
            self.error.clear()
            self.error.hide()
            axes.set_xlabel(html_to_tex(self.arg_button.descr))
        axes.set_ylabel(html_to_tex(MATERIALS_PROPERTES[param][0])
                        + ' [' +
                        html_to_tex(MATERIALS_PROPERTES[param][1]) + ']')
        self.figure.set_tight_layout(0)
        self.canvas.draw()
        warnings.showwarning = old_showwarning
        if warns:
            # if self.error.text(): self.error.append("\n")
            self.error.append("\n".join(warns))
            self.error.show()
            self.error.setFixedHeight(self.error.document().size().height())
