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

from gui.qt import QtGui, QtCore

import itertools

import numpy
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

import gui

from gui.model.materials import MATERIALS_PROPERTES, material_html_help, parse_material_components
from gui.utils.str import html_to_tex

try:
    import plask
except ImportError:
    plask = None
else:
    import plask.material


class MaterialPlot(QtGui.QWidget):

    def __init__(self, model=None, parent=None):
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

        # self.model.changed.connect(self.update_materials)

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
        material_list = [] if self.model is None else [e.name for e in self.model.entries]
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
        self.material.setMaxVisibleItems(len(material_list) - (-1 if plask is None else 1))

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
        except (ValueError, KeyError, AttributeError):
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
                if self.model is None:
                    break
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
            self.parent().setWindowTitle("Material Parameter: {} @ {}".format(param, material_name))
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


def show_material_plot(parent, model):
    # plot_window = QtGui.QDockWidget("Parameter Plot", self.document.window)
    # plot_window.setFeatures(QtGui.QDockWidget.AllDockWidgetFeatures)
    # plot_window.setFloating(True)
    # plot_window.setWidget(MaterialPlot())
    # self.document.window.addDockWidget(QtCore.Qt.BottomDockWidgetArea, plot_window)
    plot_window = QtGui.QMainWindow(parent)
    plot_window.setWindowTitle("Material Parameter")
    plot_window.setCentralWidget(MaterialPlot(model))
    plot_window.show()


def material_plot_operation(parent):
    action = QtGui.QAction(QtGui.QIcon.fromTheme('matplotlib'),
                           'Examine &Material Parameters...', parent)
    action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_M)
    action.triggered.connect(lambda: show_material_plot(parent, parent.document.materials.model))
    return action


gui.OPERATIONS.append(material_plot_operation)