from ...model.base import SectionModelTreeBased
from ...qt import QtGui
from ...qt.QtGui import QSplitter, QItemSelectionModel

from ..base import Controller
from ...utils.gui import table_last_col_fill, exception_to_msg
from ..table import table_with_manipulators
from ...model.grids.section import GridsModel
from ...utils.xml_qttree import ETreeModel


class GeometryController(Controller):

    def __init__(self, document, model=None):
        if model is None: model = SectionModelTreeBased('geometry') #TODO: native model
        Controller.__init__(self, document, model)

        self.splitter = QSplitter()
        self.tree = QtGui.QTreeView()
        self.tree.setModel(ETreeModel(model))
        self.splitter.addWidget(self.tree)

    def get_editor(self):
        return self.splitter