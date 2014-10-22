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

    #def remove_node(self):
    #    pass

    def _construct_toolbar(self):
        toolbar = QtGui.QToolBar()

        #self.remove_action = QtGui.QAction(QtGui.QIcon.fromTheme('list-remove', QtGui.QIcon(':/list-remove.png')), '&Remove', toolbar)
        #self.remove_action.setStatusTip('Remove selected node from the tree')
        # self.remove_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Minus)
        #self.remove_action.triggered.connect(self.remove_node)
        #toolbar.addAction(self.remove_action)
        #toolbar.addAction(self.tree.removeAction)

        return toolbar

    def _construct_tree(self, model):
        self.tree = QtGui.QTreeView()
        self.tree.setModel(model)
        return self.tree

    def __init__(self, document, model=None):
        if model is None: model = GeometryModel()
        Controller.__init__(self, document, model)

        external = QtGui.QGroupBox()
        vbox = QtGui.QVBoxLayout()
        external.setLayout(vbox)

        vbox.addWidget(self._construct_toolbar())
        vbox.addWidget(self._construct_tree(model))

        self.splitter = QtGui.QSplitter()
        self.splitter.addWidget(external)

    def get_widget(self):
        return self.splitter