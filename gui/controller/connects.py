from ..model.connects import ConnectsModel
from .table import TableController
from .defines import DefinesCompletionDelegate

class ConnectsController(TableController):

    def __init__(self, document, model = ConnectsModel()):
        TableController.__init__(self, document, model)
        for i in range(0, 2):
            self.table.setItemDelegateForColumn(i, DefinesCompletionDelegate(self.document.defines.model, self.table))
