from PyQt4 import QtCore

class Config(object):
    """Configuration wrapper"""

    def __init__(self):
        self.qsettings = QtCore.QSettings("plask", "gui")

    def __call__(self, key, default):
        current = self.qsettings.value(key)
        if current is None:
            self.qsettings.setValue(key, default)
            return default
        else:
            return current

    def __setitem__(self, key, value):
        self.qsettings.setValue(key, value)

CONFIG = Config()
