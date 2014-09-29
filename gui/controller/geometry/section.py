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

from ...model import SectionModelTreeBased
from ...qt import QtGui
from ...qt.QtGui import QSplitter, QItemSelectionModel

from .. import Controller
from ...utils.widgets import table_last_col_fill, exception_to_msg
from ..table import table_with_manipulators
from ...model.grids.section import GridsModel
from ...utils.xml_qttree import ETreeModel


class GeometryController(Controller):

    def __init__(self, document, model=None):
        if model is None: model = SectionModelTreeBased('geometry') #TODO: native model
        Controller.__init__(self, document, model)

        self.splitter = QSplitter()
        self.tree = QtGui.QTreeView()
        self.tree.setModel(ETreeModel(model))
        self.splitter.addWidget(self.tree)

    def get_editor(self):
        return self.splitter