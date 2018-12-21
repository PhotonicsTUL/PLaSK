# coding: utf8
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

import sys
import itertools
from copy import copy

import numpy
from lxml import etree
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor
from matplotlib.ticker import ScalarFormatter

from ...qt import QT_API
if QT_API == 'PyQt5':
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
else:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from ... import _DEBUG

from ...qt.QtCore import *
from ...qt.QtGui import *
from ...qt.QtWidgets import *
from ...model.materials import MATERIALS_PROPERTES, material_html_help, parse_material_components, MaterialsModel
from ...utils.qsignals import BlockQtSignals
from ...utils.str import html_to_tex
from ...utils.widgets import set_icon_size

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)
else:
    # 'unicode' exists, must be Python 2
    bytes = str

try:
    import plask
except ImportError:
    plask = None
else:
    import plask.material

from importlib import import_module
try:
    from importlib import reload as reload_module
except ImportError:
    reload_module = reload

PARAMS = {'T': ['300', '400']}
CURRENT_PROP = 'thermk'
CURRENT_ARG = 'T'


class MaterialPlot(QWidget):

    def __init__(self, model=None, defines=None, parent=None, init_material=None):
        super(MaterialPlot, self).__init__(parent)

        self.model = model
        self.defines = defines

        self.material = QComboBox()
        self.material.setEditable(False)
        self.material.setInsertPolicy(QComboBox.NoInsert)
        self.material.setMinimumWidth(180)
        self.material.currentIndexChanged.connect(self.material_changed)
        self.param = QComboBox()
        self.param.addItems([k for k in MATERIALS_PROPERTES.keys() if k != 'condtype'])
        self.param.setCurrentIndex(self.param.findText(CURRENT_PROP))
        self.param.currentIndexChanged.connect(self.property_changed)
        self.param.setMaxVisibleItems(len(MATERIALS_PROPERTES))
        self.param.highlighted.connect(lambda i:
            QToolTip.showText(QCursor.pos(), material_html_help(self.param.itemText(i)))
        )
        toolbar1 = QToolBar()
        toolbar1.setStyleSheet("QToolBar { border: 0px }")
        set_icon_size(toolbar1)
        toolbar1.addWidget(QLabel("Material: "))
        toolbar1.addWidget(self.material)
        toolbar1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        toolbar2 = QToolBar()
        toolbar2.setStyleSheet("QToolBar { border: 0px }")
        set_icon_size(toolbar2)
        toolbar2.addWidget(QLabel("Parameter: "))
        toolbar2.addWidget(self.param)
        toolbar2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.par_toolbar = QToolBar()
        self.par_toolbar.setStyleSheet("QToolBar { border: 0px }")
        self.mat_toolbar = QToolBar()
        self.mat_toolbar.setStyleSheet("QToolBar { border: 0px }")

        self.arguments = {}

        # self.model.changed.connect(self.update_materials)

        plot = QPushButton()
        plot.setText("&Plot")
        plot.pressed.connect(self.update_plot)
        plot.setDefault(True)
        plot.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.figure.set_facecolor(self.palette().color(QPalette.Background).name())
        self.canvas.updateGeometry()
        self.axes = None
        self.axes2 = None

        self.error = QTextEdit(self)
        self.error.setVisible(False)
        self.error.setReadOnly(True)
        self.error.setContentsMargins(0, 0, 0, 0)
        self.error.setFrameStyle(0)
        pal = self.error.palette()
        pal.setColor(QPalette.Base, QColor("#ffc"))
        self.error.setPalette(pal)
        self.error.acceptRichText()
        self.error.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox1.addWidget(toolbar1)
        hbox1.addWidget(self.mat_toolbar)
        hbox2.addWidget(toolbar2)
        hbox2.addWidget(self.par_toolbar)

        hbox2.addWidget(plot)

        layout = QVBoxLayout()
        layout.addLayout(hbox1)
        layout.addLayout(hbox2)

        self.label = QLabel(self)
        self.label.setText(' ')

        plotbox = QVBoxLayout()
        plotbox.addWidget(self.error)
        plotbox.addWidget(self.label)
        plotbox.setAlignment(self.label, Qt.AlignRight)
        plotbox.addWidget(self.canvas)

        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Vertical)
        plotbox_widget = QWidget()
        plotbox_widget.setLayout(plotbox)
        splitter.addWidget(plotbox_widget)

        self.logx_action = QAction("Logarithmic &Argument", self.canvas)
        self.logx_action.setCheckable(True)
        self.logx_action.triggered.connect(self.update_scale)
        self.logy_action = QAction("Logarithmic &Value", self.canvas)
        self.logy_action.setCheckable(True)
        self.logy_action.triggered.connect(self.update_scale)
        self.canvas.addAction(self.logy_action)
        self.canvas.addAction(self.logx_action)
        self.canvas.setContextMenuPolicy(Qt.ActionsContextMenu)

        self.info = QTextEdit(self)
        self.info.setAcceptRichText(True)
        self.info.setReadOnly(True)
        self.info.setContentsMargins(0, 0, 0, 0)
        self.info.setFrameStyle(0)
        self.info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        splitter.addWidget(self.info)

        layout.addWidget(splitter)
        self.setLayout(layout)

        self.update_materials()
        if init_material is not None:
            self.set_material(init_material, True)
            self.material.setDisabled(True)

        self.property_changed()
        for arg in self.arguments:
            arg_text = arg.text()[:-1]
            if arg_text == CURRENT_ARG:
                arg.setChecked(True)
                break
            elif arg_text == 'T':
                arg.setChecked(True)

    def resizeEvent(self, event):
        if self.error.isVisible():
            self.error.setFixedHeight(self.error.document().size().height())
        if self.info.isVisible():
            self.info.setMaximumHeight(self.info.document().size().height())
        if self.axes is not None:
            try:
                self.figure.tight_layout(pad=0.2)
            except:
                pass

    def _get_xpl_content(self):
        data = '<plask loglevel="error">\n\n'
        for m in (self.defines, self.model):
            try:
                element = m.get_file_xml_element()
            except:
                pass
            else:
                if len(element) or element.text:
                    data += etree.tostring(element, encoding='unicode', pretty_print=True) + '\n'
        data += '</plask>\n'
        return data

    def update_materials(self, *args, **kwargs):
        text = self.material.currentText()
        material_list = [] if self.model is None else [e.name for e in self.model.entries
                                                       if not isinstance(e, MaterialsModel.External)]
        sep = len(material_list)
        if plask:
            if self.model is not None:
                externals = [e for e in self.model.entries if isinstance(e, MaterialsModel.External)]
                for ext in externals:
                    if ext.what == 'library':
                        try:
                            plask.material.db.load(ext.name)
                        except RuntimeError:
                            pass
                    elif ext.what == 'module':
                        sys.path.insert(0, '.')
                        try:
                            if ext.name in sys.modules:
                                reload_module(sys.modules[ext.name])
                            else:
                                import_module(ext.name)
                        except:
                            pass
                        finally:
                            sys.path = sys.path[1:]
            self.materialdb = copy(plask.material.db)
            self.manager = plask.Manager(self.materialdb, draft=True)
            self.manager.load(self._get_xpl_content())
            material_blacklist = ['dielectric', 'liquid_crystal', 'metal', 'semiconductor', 'air']
            material_list.extend(sorted((mat for mat in plask.material.db
                                         if mat not in material_list and mat not in material_blacklist),
                                        key=lambda x: x.lower()))
        self.material.clear()
        self.material.addItems(material_list)
        self.material.insertSeparator(sep)
        if args:
            self.material.setEditText(text)
        self.material.setMaxVisibleItems(len(material_list) - (-1 if plask is None else 1))

    def param_changed(self, value):
        name = self.sender().objectName()
        name, i = name[:-1], int(name[-1])
        PARAMS.setdefault(name, ['', ''])[i] = value

    def set_toolbar(self, toolbar, values, what):
        """
        :param what: 0: component, 1: doping, 2: property argument
        """
        for name, descr, unit in values:
            if name is None:
                toolbar.addSeparator()
                continue
            select = QRadioButton()
            select.toggled.connect(self.selected_argument)
            select.setAutoExclusive(False)
            toolbar.addWidget(select)
            select.setText("{}:".format(name))
            select.descr = descr
            select.unit = unit
            val0 = QLineEdit()
            val0.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            val0.returnPressed.connect(self.update_plot)
            val0.setObjectName(name+'0')
            val0.textChanged.connect(self.param_changed)
            toolbar.addWidget(val0)
            sep = toolbar.addWidget(QLabel("-"))
            sep.setVisible(False)
            val1 = QLineEdit()
            val1.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            val1.returnPressed.connect(self.update_plot)
            val1.setObjectName(name+'1')
            val1.textChanged.connect(self.param_changed)
            select_text = 'doping' if what == 1 else name
            if select_text in PARAMS:
                val0.setText(PARAMS[select_text][0])
                val1.setText(PARAMS[select_text][1])
            act2 = toolbar.addWidget(val1)
            act2.setVisible(False)
            self.arguments[select] = val0, val1, (sep, act2), what
        for arg in self.arguments:
            if arg.isChecked(): return
            arg_text = arg.text()[:-1]
            if arg_text == CURRENT_ARG:
                arg.setChecked(True)
                return

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
                    value = ", ".join(str(v) for v in value)
                text += "<tr><td><b>{}: </b></td><td>{}</td></tr>".format(key, value)
            self.info.setText(text + "</table>")
            self.info.resize(self.info.document().idealWidth(), self.info.document().size().height())
            self.info.show()
            self.info.setMaximumHeight(self.info.document().size().height())
        else:
            self.info.hide()

    def set_material(self, material, update_toolbar=False):
        for child in self.mat_toolbar.children():
            if child in self.arguments:
                del self.arguments[child]
        self.mat_toolbar.clear()

        entries = [e for e in self.model.entries if e.name == material]
        if entries: alloy = entries[0].alloy
        else: alloy = None
        name, label, groups, dope = parse_material_components(material, alloy)

        if groups:
            elements = tuple(itertools.chain(*([e[0] for e in g] + [None] for g in groups if len(g) > 1)))[:-1]
            if update_toolbar:
                PARAMS.update(
                    dict(itertools.chain(*([(e[0], [e[1],'']) for e in g if e[1]] for g in groups if len(g) > 1))))
            self.set_toolbar(self.mat_toolbar, ((e, "{} fraction".format(e), "-") for e in elements), 0)
            name = ''.join(itertools.chain(*([e[0] for e in g] for g in groups)))
        else:
            elements = None

        if dope is not None:
            dopes = dope.split('=')
            dope = dopes[0]
            if ' ' in dope: dope = dope.split()[0]
            if elements:
                self.mat_toolbar.addSeparator()
            if len(dopes) > 1 and update_toolbar:
                PARAMS['doping'] = dopes[1], ''
            self.set_toolbar(self.mat_toolbar,
                             [(("[" + dope + "]"), dope + " doping concentration", "cm<sup>-3</sup>")], 1)

        if update_toolbar:
            if label: name += '_' + label
            if dope is not None: name += ':' + dope
            i = self.material.findText(name)
            if i == -1:
                i = self.material.count() + 1
                self.material.insertSeparator(i-1)
                self.material.addItem(name)
            with BlockQtSignals(self.material):
                self.material.setCurrentIndex(i)

        self.update_info()

    def material_changed(self):
        self.set_material(self.material.currentText())

    def property_changed(self):
        for child in self.par_toolbar.children():
            if child in self.arguments:
                del self.arguments[child]
        self.par_toolbar.clear()
        self.set_toolbar(self.par_toolbar, MATERIALS_PROPERTES[self.param.currentText()][2], 2)
        global CURRENT_PROP
        CURRENT_PROP = self.param.currentText()
        self.update_info()

    def selected_argument(self):
        button = self.sender()
        checked = button.isChecked()
        for act in self.arguments[button][2]:
            act.setVisible(checked)
        if checked:
            global CURRENT_ARG
            CURRENT_ARG = button.text()[:-1]
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
                try:
                    val = float(val)
                except ValueError:
                    val = str(val)
                if cat == 1:
                    yield 'doping', val
                else:
                    yield str(k.text())[:-1].replace('&', ''), val

    def update_plot(self):
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.axes2 = None
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
            plot_range = numpy.linspace(start, end, 1001)
            plot_cat = self.arguments[self.arg_button][3]
            other_args = dict(self._parse_other_args(self.arg_button, 2))
            other_elements = dict(self._parse_other_args(self.arg_button, 0))
            other_elements.update(dict(('doping', v) for k,v in self._parse_other_args(self.arg_button, 1)))
            arg_name = 'doping' if plot_cat == 1 else str(self.arg_button.text())[:-1].replace('&', '')
            material_name = str(self.material.currentText())
            if plask is not None:
                if plot_cat == 2:
                    material = self.materialdb.get(material_name, **other_elements)
                    self.vals = lambda a: material.__getattribute__(param)(**dict(((arg_name, a),), **other_args))
                else:
                    self.vals = lambda a: self.materialdb.get(material_name, **dict(((arg_name, a),), **other_elements)).\
                        __getattribute__(param)(**other_args)
            else:
                model_materials = set()
                material = None
                if self.model is not None:
                    while material_name not in model_materials:  # loop for looking-up the base
                        material = [e for e in self.model.entries if e.name == material_name]
                        if material:
                            model_materials.add(material_name)
                            material = material[0]
                            mprop = [p for p in material.properties if p[0] == param]
                            if mprop:
                                expr = mprop[0][1]
                                code = compile(expr, '', 'eval')
                                class Material(object): pass
                                mat = Material()
                                for k, v in other_elements.items():
                                    setattr(mat, k, v)
                                other_args['self'] = mat
                                if plot_cat == 2:
                                    self.vals = lambda a: eval(code, numpy.__dict__,
                                                               dict(((arg_name, a),), **other_args))
                                else:
                                    def f(a):
                                        setattr(mat, arg_name, a)
                                        return eval(code, numpy.__dict__, dict((), **other_args))
                                    self.vals = f
                                break
                            else:
                                material_name = material.base  # and we repeat the loop
                                material = None
                        else:
                            break
            vals = numpy.array([self.vals(a) for a in plot_range])
            if vals.dtype == complex:
                self.axes2 = self.axes.twinx()
                self.axes.plot(plot_range, vals.real)
                self.axes2.plot(plot_range, vals.imag, ls='--')
            else:
                self.axes.plot(plot_range, vals)
            self.parent().setWindowTitle("Material Parameter: {} @ {}".format(param, material_name))
        except Exception as err:
            if _DEBUG:
                import traceback
                traceback.print_exc()
            self.error.setText('<div style="color:red;">{}</div>'.format(str(err)))
            self.error.show()
            self.label.hide()
            self.error.setFixedHeight(self.error.document().size().height())
        else:
            self.error.clear()
            self.error.hide()
            self.xn = self.arg_button.text()[:-1].replace('&', '')
            self.yn = param
            self.xu = self.arg_button.unit
            self.yu = MATERIALS_PROPERTES[param][1]
            self.label.show()
            self.label.setText(' ')
            self.axes.set_xlim(start, end)
            self.axes.set_xlabel(html_to_tex(u"{}{} [{}]".format(self.arg_button.descr[0].upper(),
                                                                 self.arg_button.descr[1:],
                                                                 self.arg_button.unit)))
            self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
            self.axes.set_ylabel('[]')
            if self.axes2 is not None:
                self.axes2.set_ylabel('[]')
            self.figure.tight_layout(pad=0.2)
            label = html_to_tex(MATERIALS_PROPERTES[param][0]).splitlines()[0] +\
                    ' [' + html_to_tex(MATERIALS_PROPERTES[param][1]) + ']'
            if self.axes2 is None:
                self.axes.set_ylabel(label)
            else:
                self.axes.set_ylabel(label + " (real part, solid)")
                self.axes2.set_ylabel(label + " (imaginary part, dashed)")
            self._cursor = Cursor(self.axes, horizOn=False, useblit=True, color='#888888', linewidth=1)
            self.update_scale()
            warnings.showwarning = old_showwarning
            if warns:
                # if self.error.text(): self.error.append("\n")
                self.error.append("\n".join(warns))
                self.error.show()
                self.error.setFixedHeight(self.error.document().size().height())

    def update_scale(self):
        fmtr = ScalarFormatter(useOffset=False)
        if self.axes is not None:
            self.axes.set_xscale('log' if self.logx_action.isChecked() else 'linear')
            self.axes.set_yscale('log' if self.logy_action.isChecked() else 'linear')
            self.axes.get_xaxis().set_major_formatter(fmtr)
            self.axes.get_yaxis().set_major_formatter(fmtr)
        if self.axes2 is not None:
            self.axes2.set_xscale('log' if self.logx_action.isChecked() else 'linear')
            self.axes2.set_yscale('log' if self.logy_action.isChecked() else 'linear')
            self.axes2.get_yaxis().set_major_formatter(fmtr)
        self.figure.tight_layout(pad=0.2)
        self.canvas.draw()

    def on_mouse_move(self, event):
        if not self.label.isVisible(): return
        if not event.inaxes:
            self.label.setText(' ')
            return
        x = event.xdata
        y = self.vals(x)
        if isinstance(y, tuple):
            y = '(' + ', '.join("{:.5g}".format(i) for i in y) + ')'
        else:
            y = "{:.5g}".format(y)
        xu = '' if self.xu == '-' else ' ' + self.xu
        yu = '' if self.yu == '-' else ' ' + self.yu
        self.label.setText("{self.xn} = {x:.5g}{xu}    {self.yn} = {y}{yu}".format(**locals()))


def show_material_plot(parent, model, defines, init_material=None):
    # plot_window = QDockWidget("Parameter Plot", self.document.window)
    # plot_window.setFeatures(QDockWidget.AllDockWidgetFeatures)
    # plot_window.setFloating(True)
    # plot_window.setWidget(MaterialPlot())
    # self.document.window.addDockWidget(Qt.BottomDockWidgetArea, plot_window)
    try:
        plot = MaterialPlot(model, defines, init_material=init_material)
    except Exception as err:
        QMessageBox.critical(None, "Error while parsing materials", str(err))
        return
    plot_window = QMainWindow(parent)
    plot_window.setWindowTitle("Material Parameter")
    plot_window.setCentralWidget(plot)
    plot_window.show()
