from PyQt4 import QtCore 
from PyQt4 import QtGui

def showError(msg, parent = None, title = None):
    err = QtGui.QErrorMessage(parent)
    err.setModal(True)
    if title != None: err.setWindowTitle(title)
    err.showMessage(msg)

def exceptionToMsg(callable, parent = None, err_title = None):
    try:
        callable()
        return True
    except Exception as e:
        showError(str(e), parent, err_title)
        return False

#deprecated, to remove
def toPythonStr(qstr):
    if isinstance(qstr, QtCore.QString):
        return unicode (qstr.toUtf8(), "utf-8")
    else:
        return qstr
