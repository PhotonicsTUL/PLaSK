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

from ...qt import QtCore, QtGui, qt
from ...qt.QtCore import Qt

from ...model.script.completer import CompletionsModel, get_completions


class CompletionsPopup(QtGui.QTableView):

    def __init__(self, textedit):
        super(CompletionsPopup, self).__init__(None)
        self.setMinimumHeight(150)
        self._textedit = textedit
        # self.setAlternatingRowColors(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setSortingEnabled(False)
        self.setShowGrid(False)
        self.setWordWrap(False)
