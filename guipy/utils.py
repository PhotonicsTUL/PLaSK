from PyQt4 import QtCore 

def toPythonStr(qstr):
    if isinstance(qstr, QtCore.QString):
        return unicode (qstr.toUtf8(), "utf-8")
    else:
        return qstr
