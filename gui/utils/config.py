# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from ..qt import QtCore

_parsed = {'true': True, 'yes': True, 'false': False, 'no': False}

def parse_highlight(string):
    """Parse syntax highlighting from config"""
    result = {}
    for item in string.split(','):
        item = item.strip()
        key, val = item.split('=')
        result[key] = _parsed.get(val, val)
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
            try:
                return _parsed.get(current, current)
            except TypeError:
                return current

    def __getitem__(self, key):
        value = self.qsettings.value(key)
        try:
            return _parsed.get(value, value)
        except TypeError:
            return value

    def __setitem__(self, key, value):
        self.qsettings.setValue(key, value)

    def sync(self):
        """Synchronize settings"""
        self.qsettings.sync()

CONFIG = Config()
