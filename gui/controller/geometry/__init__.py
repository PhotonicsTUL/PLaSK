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

from ...model.geometry import GeometryModel
from ...qt import QtGui

from .. import Controller


class GeometryController(Controller):

    def __init__(self, document, model=None):
        if model is None: model = GeometryModel()
        Controller.__init__(self, document, model)

        self.splitter = QtGui.QSplitter()
        self.tree = QtGui.QTreeView()
        self.tree.setModel(model)
        self.splitter.addWidget(self.tree)

    def get_widget(self):
        return self.splitter