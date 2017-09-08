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

import weakref

from ..qt.QtCore import *

from ..qt.QtCore import *
from ..qt.QtGui import *
from ..utils.signal import Signal


class Info(object):

    NONE = 0
    GROUP = 1
    INFO = 2
    WARNING = 3
    ERROR = 4

    def __str__(self):
        return self.text

    def __init__(self, text, level=NONE, **kwargs):
        object.__init__(self)
        self.text = text
        self.level = int(level)
        for name, value in kwargs.items():
            setattr(self, name, value)

    def add_connection(self, attr_name, value):
        getattr(self, attr_name, []).append(value)

    def has_connection(self, attr_name, value, ans_if_non_attr=True):
        """
            Check if self has attribute with given name which includes given value.
            For example: self.has_connection('rows', 5)
            :param attr_name: required attribute
            :param value: required value
            :param ans_if_non_attr: result which is returned when object has no required attribute
            :return: True if self has attribute with given name which includes given value,
                     False if self has attribute with given name which dpesn't include given value,
                     ans_if_non_attr if self has not attribute with given name.
        """
        #if hasattr(self, attr_name): return getattr(self, attr_name) == value
        if hasattr(self, attr_name): return value in getattr(self, attr_name)   # + 's'
        return ans_if_non_attr


def info_level_icon(level):
    if level == Info.GROUP: return QIcon.fromTheme('folder')
    if level == Info.INFO: return QIcon.fromTheme('dialog-information')
    if level == Info.WARNING: return QIcon.fromTheme('dialog-warning')
    if level == Info.ERROR: return QIcon.fromTheme('dialog-error')
    return None


class InfoListModel(QAbstractListModel):
    """
    Qt list model of info (warning, errors, etc.) of section model
    (None section model is allowed and than the list is empty)
    """

    def _set_model(self, model):
        if hasattr(self, 'model'):
            m = self.model()
            if m: m.infoChanged -= self.infoChanged
        if model is None:
            if hasattr(self, 'model'): del self.model
            self.entries = []
        else:
            self.model = weakref.ref(model)
            self.entries = model.get_info()
            model.infoChanged += self.infoChanged

    def __init__(self, model, parent=None, *args):
        QAbstractListModel.__init__(self, parent, *args)
        self._set_model(model)

    def infoChanged(self, model, *args, **kwargs):
        """Read info from model, inform observers."""
        self.layoutAboutToBeChanged.emit()
        self.entries = model.get_info()
        self.layoutChanged.emit()

    def setModel(self, model):
        self._set_model(model)
        if model is not None: self.infoChanged(model)

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid(): return 0
        return len(self.entries)

    #def columnCount(self, parent=QModelIndex()):
    #    return 1

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid(): return None
        if role == Qt.DisplayRole:
            row = index.row()
            n = len(self.entries)
            if row == 0 and n > 1:
                return self.entries[row].text + "  (+{} message{})".format(n-1, "" if n == 2 else "s")
            else:
                return self.entries[row].text
        if role == Qt.DecorationRole:
            return info_level_icon(self.entries[index.row()].level)
        if role == Qt.TextAlignmentRole:
            try: return self.entries[index.row()].align
            except AttributeError: return None
        # if role == Qt.FontRole:
        #     font = QFont()
        #     font.setUnderline(True)
        #     return font
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return 'text'
        return None


class InfoSource(object):

    def __init__(self, info_cb=None):
        """
            :param info_cb: call when list of errors has been changed with parameters: section name, list of errors
        """
        object.__init__(self)
        self._info = []      # model Infos: Errors, Warnings and Informations
        self.infoChanged = Signal()
        if info_cb: self.infoChanged.connect(info_cb)

    def mark_info_invalid(self):
        """Invalidate the info and it indexes."""
        self._info = None

    def fire_info_changed(self):
        """
            Inform observers that info has been changed.
            You must call mark_info_invalid and prepare data to build the info before call this.
        """
        self.infoChanged(self)

    def create_info(self, *args, **kwargs):
        """
            Create table with messages.
            :return: array of Info objects
        """
        return []

    def get_info(self, level=None):
        """
            Get array of Info objects on given level connected with this object.
        """
        if self._info is None:
            self._info = self.create_info()
        if level is not None:
            return tuple(m for m in self._info if m.level == level)
        else:
            return self._info

    #@property
    #def info(self):
    #    return self.get_info()

    def get_list_model(self):
        return InfoListModel(self)

    def refresh_info(self):
        self._info = None
        self.infoChanged(self)
