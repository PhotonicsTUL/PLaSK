from model.connects import ConnectsModel
from controler.table import TableControler

class ConnectsControler(TableControler):

    def __init__(self, document, model = ConnectsModel()):
        TableControler.__init__(self, document, model)        
