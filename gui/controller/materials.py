import re
import itertools

from ..qt import QtCore, QtGui, qt
from ..qt.QtGui import QSplitter

from ..utils.str import html_to_tex

from ..model.materials import MaterialsModel, MaterialPropertyModel, material_html_help, \
                              MATERIALS_PROPERTES, ELEMENT_GROUPS
from ..utils.gui import HTMLDelegate, table_last_col_fill
from .base import Controller
from .defines import DefinesCompletionDelegate
from .table import table_with_manipulators

try:
    import matplotlib
except ImportError:
    matplotlib = None
else:
    import numpy
    matplotlib.rc('backend', qt4=qt)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

try:
    import plask
except ImportError:
    plask = None
else:
    import plask.material


class MaterialBaseDelegate(DefinesCompletionDelegate):

    @staticmethod
    def _format_material(mat):
        return mat

    def __init__(self, defines_model, parent):
        DefinesCompletionDelegate.__init__(self, defines_model, parent)

    def createEditor(self, parent, option, index):

        earlier_names = ['dielectric', 'liquid_crystal', 'metal', 'semiconductor']

        if plask:
            earlier_names.extend(
                sorted((self._format_material(mat) for mat in plask.material.db if mat not in earlier_names),
                       key=lambda x: x.lower()))

        earlier_names.extend(e.name for e in index.model().entries[0:index.row()])

        if not earlier_names: return super(MaterialBaseDelegate, self).createEditor(parent, option, index)

        combo = QtGui.QComboBox(parent)
        combo.setEditable(True)
        combo.setInsertPolicy(QtGui.QComboBox.NoInsert)
        combo.addItems(earlier_names)
        combo.insertSeparator(4)
        combo.insertSeparator(len(earlier_names)-index.row()+1)
        combo.setEditText(index.data())
        combo.setCompleter(self.get_defines_completer(parent))
        combo.setMaxVisibleItems(len(earlier_names))
        #self.connect(combo, QtCore.SIGNAL("currentIndexChanged(int)"),
        #             self, QtCore.SLOT("currentIndexChanged()"))
        return combo


class MaterialPropertiesDelegate(DefinesCompletionDelegate):

    def __init__(self, defines_model, parent):
        DefinesCompletionDelegate.__init__(self, defines_model, parent)

    def createEditor(self, parent, option, index):
        opts = index.model().options_to_choose(index)

        if opts is None: return super(MaterialPropertiesDelegate, self).createEditor(parent, option, index)

        combo = QtGui.QComboBox(parent)
        combo.setEditable(True)
        combo.setInsertPolicy(QtGui.QComboBox.NoInsert)
        combo.addItems(opts)
        combo.setEditText(index.data())
        completer = combo.completer()
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitive)
        combo.setCompleter(completer)
        combo.highlighted.connect(lambda i:
            QtGui.QToolTip.showText(QtGui.QCursor.pos(), material_html_help(combo.itemText(i)))
        )
        combo.setMaxVisibleItems(len(opts))
        #combo.setCompleter(completer)
        #self.connect(combo, QtCore.SIGNAL("currentIndexChanged(int)"),
        #             self, QtCore.SLOT("currentIndexChanged()"))
        return combo


class MaterialsController(Controller):

    def __init__(self, document, selection_model=None):
        if selection_model is None: selection_model = MaterialsModel()
        Controller.__init__(self, document, selection_model)

        self.splitter = QSplitter()

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


elements_re = re.compile("[A-Z][a-z]*")


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
        self.property_changed()

        self.model.changed.connect(self.update_materials)
        self.update_materials()

        plot = QtGui.QPushButton()
        plot.setText("&Plot")
        plot.pressed.connect(self.update_plot)
        plot.setDefault(True)
        plot.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        self.error = QtGui.QTextEdit()
        self.error.setVisible(False)
        self.error.setReadOnly(True)
        self.error.setContentsMargins(0,0,0,0)
        self.error.setFrameStyle(0)
        pal = self.error.palette();
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#FFFFCC"))
        self.error.setPalette(pal)
        self.error.acceptRichText()

        hbox1 = QtGui.QHBoxLayout()
        hbox2 = QtGui.QHBoxLayout()
        hbox1.addWidget(toolbar1)
        hbox1.addWidget(self.mat_toolbar)
        hbox2.addWidget(toolbar2)
        hbox2.addWidget(self.par_toolbar)

        hbox2.addWidget(plot)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addWidget(self.error)
        vbox.addWidget(self.canvas)

        self.setLayout(vbox)

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

    @staticmethod
    def parse_material(material):
        if plask:
            try:
                simple = plask.material.db.is_simple(str(material))
            except ValueError:
                simple = True
            except RuntimeError:
                simple = True
        else:
            simple = True
        if ':' in material:
            name, doping = material.split(':')
        else:
            name = material
            doping = None
        return doping, name, simple

    def material_changed(self):
        old = dict((k.text(), (v[0].text(), v[1].text())) for k,v in self.arguments.items())
        for child in self.mat_toolbar.children():
            if child in self.arguments:
                del self.arguments[child]
        self.mat_toolbar.clear()
        material = self.material.currentText()

        dope, name, simple = self.parse_material(material)

        if not simple:
            elements = elements_re.findall(name)
            groups = ([e for e in elements if e in g][:-1] for g in ELEMENT_GROUPS)
            elements = tuple(itertools.chain(*(g + ([None] if g else []) for g in groups)))[:-1]
            self.set_toolbar(self.mat_toolbar, ((e, "{} fraction".format(e)) for e in elements), old, 0)
        else:
            elements = None

        if dope is not None:
            if elements:
                self.mat_toolbar.addSeparator()
            self.set_toolbar(self.mat_toolbar,
                             [("["+dope+"]", dope + "doping concentration [1/cm<sup>3</sup>]")], old, 1)

    def property_changed(self):
        old = dict((k.text(), (v[0].text(), v[1].text())) for k,v in self.arguments.items())
        for child in self.par_toolbar.children():
            if child in self.arguments:
                del self.arguments[child]
        self.par_toolbar.clear()
        self.set_toolbar(self.par_toolbar, MATERIALS_PROPERTES[self.param.currentText()][2], old, 2)

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
        else:
            self.error.clear()
            self.error.hide()
            axes.set_xlabel(html_to_tex(self.arg_button.descr))
        axes.set_ylabel(html_to_tex(MATERIALS_PROPERTES[param][0])
                        + ' [' +
                        html_to_tex(MATERIALS_PROPERTES[param][1]) + ']')
        self.figure.set_tight_layout(5)
        self.canvas.draw()
        warnings.showwarning = old_showwarning
        if warns:
            # if self.error.text(): self.error.append("\n")
            self.error.append("\n".join(warns))
            self.error.show()
