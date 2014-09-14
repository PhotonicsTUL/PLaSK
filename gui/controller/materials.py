import re

from ..qt import QtCore, QtGui, qt
from ..qt.QtGui import QSplitter

from ..model.materials import MaterialsModel, MaterialPropertyModel, materialHTMLHelp, MATERIALS_PROPERTES
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
        return _mat

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
            QtGui.QToolTip.showText(QtGui.QCursor.pos(), materialHTMLHelp(combo.itemText(i)))
        )
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


i_re = re.compile("<i>(.*?)</i>")
sub_re = re.compile("<sub>(.*?)</sub>")
sup_re = re.compile("<sup>(.*?)</sup>")


def _html_to_tex(s):
    '''Poor man's HTML to MathText conversion'''
    s = s.replace(" ", "\/")
    s = i_re.sub(r"\mathit{\g<1>}", s)
    s = sub_re.sub(r"_{\g<1>}", s)
    s = sup_re.sub(r"^{\g<1>}", s)
    return r"$\sf " + s + "$"


class MaterialPlot(QtGui.QWidget):

    def __init__(self, model, parent=None):
        super(MaterialPlot, self).__init__(parent)
        #self.setContentsMargins(0, 0, 0, 0)

        self.model = model

        self.material = QtGui.QComboBox()
        self.material.setEditable(True)
        self.material.setInsertPolicy(QtGui.QComboBox.NoInsert)
        self.material.setMinimumWidth(180)
        #self.material.changed.connect(self.update_material)
        self.param = QtGui.QComboBox()
        self.param.addItems([k for k in MATERIALS_PROPERTES.keys() if k != 'condtype'])
        self.param.currentIndexChanged.connect(self.update_vars)
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
        self.update_vars()

        self.model.changed.connect(self.update_materials)
        self.update_materials()

        plot = QtGui.QPushButton()
        plot.setText("&Plot")
        plot.pressed.connect(self.update_plot)
        plot.setDefault(True)

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
        hbox1.addWidget(plot)
        hbox2.addWidget(toolbar2)
        hbox2.addWidget(self.par_toolbar)

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

    def update_material(self):
        pass

    def update_vars(self):
        self.par_toolbar.clear()
        first = True
        old = dict((k.text(), (v[0].text(), v[1].text())) for k,v in self.arguments.items())
        self.arguments = {}
        for v in MATERIALS_PROPERTES[self.param.currentText()][2]:
            select = QtGui.QRadioButton()
            select.toggled.connect(self.update_arg)
            self.par_toolbar.addWidget(select)
            select.setText("{}:".format(v[0]))
            select.descr = v[1]
            val1 = QtGui.QLineEdit()
            self.par_toolbar.addWidget(val1)
            sep = self.par_toolbar.addWidget(QtGui.QLabel("-"))
            val2 = QtGui.QLineEdit()
            if select.text() in old:
                val1.setText(old[select.text()][0])
                val2.setText(old[select.text()][1])
            act2 = self.par_toolbar.addWidget(val2)
            self.arguments[select] = val1, val2, (sep, act2)
            if first:
                select.setChecked(True)
                first = False
            else:
                sep.setVisible(False)
                act2.setVisible(False)

    def update_arg(self):
        button = self.sender()
        checked = button.isChecked()
        for act in self.arguments[button][2]:
            act.setVisible(checked)
        if checked:
            self.arg_button = button

    def _parse_other_args(self, button):
        for i in self.arguments.items():
            val = i[1][0].text()
            if i[0] is not button and val:
                try: val = float(val)
                except ValueError: pass
                yield str(i[0].text())[:-1], val

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
                arg1, arg2 = (float(v.text()) for v in self.arguments[self.arg_button][:2])
            except ValueError:
                raise ValueError("Wrong ranges '{}' - '{}'"
                                 .format(*(v.text() for v in self.arguments[self.arg_button][:2])))
            other_args = dict(self._parse_other_args(self.arg_button))
            args = plask.linspace(arg1, arg2, 1000)
            argn = str(self.arg_button.text())[:-1]
            matn = str(self.material.currentText())
            while True:
                material = [e for e in self.model.entries if e.name == matn]
                if material:
                    material = material[0]
                    mprop = [p for p in material.properties if p[0] == param]
                    if mprop:
                        expr = mprop[0][1]
                        code = compile(expr, '', 'eval')
                        vals = [eval(code, numpy.__dict__, dict(((argn, a),), **other_args)) for a in args]
                        break
                    else:
                        matn = material.base
                        material = None
                else:
                    break
            if not material:
                material = plask.material.db.get(matn)
                vals = [material.__getattribute__(param)(**dict(((argn, a),), **other_args)) for a in args]
            axes.plot(args, vals)
        except Exception as err:
            self.error.setText('<div style="color:red;">{}</div>'.format(str(err)))
            self.error.show()
        else:
            self.error.clear()
            self.error.hide()
            axes.set_xlabel(_html_to_tex(self.arg_button.descr))
        axes.set_ylabel(_html_to_tex(MATERIALS_PROPERTES[param][0])
                        + ' [' + _html_to_tex(MATERIALS_PROPERTES[param][1]) + ']')
        self.figure.set_tight_layout(5)
        self.canvas.draw()
        warnings.showwarning = old_showwarning
        if warns:
            # if self.error.text(): self.error.append("\n")
            self.error.append("\n".join(warns))
            self.error.show()
