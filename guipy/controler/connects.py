from model.connects import ConnectsModel
from controler.table import TableControler
from controler.defines import DefinesCompletionDelegate

class ConnectsControler(TableControler):

    def __init__(self, document, model = ConnectsModel()):
        TableControler.__init__(self, document, model)        
        for i in range(0, 2):
            self.table.setItemDelegateForColumn(i, DefinesCompletionDelegate(self.document.defines.model, self.table))