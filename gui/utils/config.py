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
