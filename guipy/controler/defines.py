from model.defines import DefinesModel
from controler.table import TableControler

class DefinesControler(TableControler):

    def __init__(self, document, model = DefinesModel()):
        TableControler.__init__(self, document, model)        
