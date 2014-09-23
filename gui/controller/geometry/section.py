from ...qt import QtGui
from ...qt.QtGui import QSplitter, QItemSelectionModel

from ..base import Controller
from ...utils.gui import table_last_col_fill, exception_to_msg
from ..table import table_with_manipulators
from ...model.grids.section import GridsModel

class GeometryController(Controller):

    def __init__(self, document, model=None):
        #if model is None: model = GridsModel()
        Controller.__init__(self, document, model)