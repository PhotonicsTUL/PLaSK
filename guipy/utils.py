# needs by Signal:
from __future__ import print_function
import inspect
from weakref import WeakSet, WeakKeyDictionary


from PyQt4 import QtCore 
from PyQt4 import QtGui


def showError(msg, parent = None, title = None):
    err = QtGui.QErrorMessage(parent)
    err.setModal(True)
    if title != None: err.setWindowTitle(title)
    err.showMessage(msg)

def exceptionToMsg(f, parent = None, err_title = None):
    try:
        f()
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
    
defaultFont = QtGui.QFont()
defaultFont.setFamily("Courier")
defaultFont.setPointSize(10)

""" A signal/slot implementation + __iadd__ and __isub__ (Piotr Beling)

Author:  Thiago Marcos P. Santos
Author:  Christopher S. Case
Author:  David H. Bronke
Created: August 28, 2008
Updated: December 12, 2011
License: MIT

"""
class Signal(object):
    def __init__(self):
        self._functions = WeakSet()
        self._methods = WeakKeyDictionary()

    def __call__(self, *args, **kargs):
        # Call handler functions
        for func in self._functions:
            func(*args, **kargs)

        # Call handler methods
        for obj, funcs in self._methods.items():
            for func in funcs:
                func(obj, *args, **kargs)

    def connect(self, slot):
        if inspect.ismethod(slot):
            if slot.__self__ not in self._methods:
                self._methods[slot.__self__] = set()

            self._methods[slot.__self__].add(slot.__func__)

        else:
            self._functions.add(slot)
            
    def __iadd__(self, slot):
        self.connect(slot)
        return self

    def disconnect(self, slot):
        if inspect.ismethod(slot):
            if slot.__self__ in self._methods:
                self._methods[slot.__self__].remove(slot.__func__)
        else:
            if slot in self._functions:
                self._functions.remove(slot)
                
    def __isub__(self, slot):
        self.disconnect(slot)
        return self

    def clear(self):
        self._functions.clear()
        self._methods.clear()