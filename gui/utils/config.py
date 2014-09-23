from ..qt import QtCore


def parse_highlight(string):
    """Parse syntax highlighting from config"""
    result = {}
    for item in string.split(','):
        item = item.strip()
        key, val = item.split('=')
        if val.lower() in ('true', 'yes'): val = True
        elif val.lower() in ('false', 'no'): val = False
        result[key] = val
    return result


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

    def __getitem__(self, key):
        return self.qsettings.value(key)

    def __setitem__(self, key, value):
        self.qsettings.setValue(key, value)

    def sync(self):
        """Synchronize settings"""
        self.qsettings.sync()

CONFIG = Config()
