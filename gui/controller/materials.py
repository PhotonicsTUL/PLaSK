from ..qt import QtCore, QtGui
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
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

try:
    import plask
except ImportError:
    plask = None


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
        plot_window.setCentralWidget(MaterialPlot())
        plot_window.show()



class MaterialPlot(QtGui.QWidget):

    def __init__(self, parent=None):
        super(MaterialPlot, self).__init__(parent)
        #self.setContentsMargins(0, 0, 0, 0)

        material_list = []
        material_blacklist = ['dielectric', 'liquid_crystal', 'metal', 'semiconductor', 'air']
        if plask:
            material_list.extend(sorted((mat for mat in plask.material.db
                                         if mat not in material_list and mat not in material_blacklist),
                                        key=lambda x: x.lower()))
        # material_list.extend(e.name for e in index.model().entries[0:index.row()])

        self.toolbar = QtGui.QToolBar()
        self.material = QtGui.QComboBox()
        self.material.setEditable(True)
        self.material.setInsertPolicy(QtGui.QComboBox.NoInsert)
        self.material.addItems(material_list)
        self.param = QtGui.QComboBox()
        self.param.addItems([k for k in MATERIALS_PROPERTES.keys() if k != 'condtype'])
        self.param.currentIndexChanged.connect(self.update_vars)
        self.toolbar.addWidget(QtGui.QLabel("Material: "))
        self.toolbar.addWidget(self.material)
        self.toolbar.addWidget(QtGui.QLabel("  Parameter: "))
        self.toolbar.addWidget(self.param)
        self.toolbar.addWidget(QtGui.QLabel(" "))
        self.toolbar.addSeparator()
        self.vars = None
        self.update_vars()

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.figure.set_tight_layout(1)
        self.axes.hold(False)   # we want the axes cleared every time plot() is called
        canvas = FigureCanvas(self.figure)
        canvas.setParent(self)
        canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        canvas.updateGeometry()

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.toolbar)
        vbox.addWidget(canvas)

        self.setLayout(vbox)

    def update_vars(self):
        if self.vars is not None:
            self.toolbar.removeAction(self.vars)
        vars = QtGui.QWidget()
        layout = QtGui.QHBoxLayout(vars)
        layout.setContentsMargins(0, 0, 0, 0)
        vars.setLayout(layout)
        first = True
        for v in MATERIALS_PROPERTES[self.param.currentText()][2]:
            select = QtGui.QRadioButton()
            if first:
                select.setChecked(True)
                first = False
            layout.addWidget(select)
            layout.addWidget(QtGui.QLabel("{}:".format(v[0])))
            layout.addWidget(QtGui.QLineEdit())
        self.vars = self.toolbar.addWidget(vars)
