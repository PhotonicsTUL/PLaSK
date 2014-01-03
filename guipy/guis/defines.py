from PyQt4 import QtCore, QtGui
from models.defines import DefinesModel

class DefinesEditor(QtGui.QWidget):
    
    def __init__(self, parent, definesModel):
        super(DefinesEditor, self).__init__(parent)
        self.initUI()
        
    def initUI(self, definesModel):               
        table = QtGui.QTableView()
        table.setModel(definesModel)
        self.setCentralWidget(table)

        toolbar = self.addToolBar('Edit')
        #toolbar.addAction(addAction)
