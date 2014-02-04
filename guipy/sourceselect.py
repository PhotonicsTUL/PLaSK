from PyQt4 import QtGui
from PyQt4.QtGui import QDialogButtonBox

class SourceSelectDialog(QtGui.QDialog):
    
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        
        #connect(buttonBox, SIGNAL(accepted()), self, SLOT(accept()))
        #connect(buttonBox, SIGNAL(rejected()), self, SLOT(reject()))