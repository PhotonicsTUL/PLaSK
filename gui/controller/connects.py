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

from ..qt.QtWidgets import *

from .table import TableController
from .defines import DefinesCompletionDelegate

from ..model.connects import ConnectsModel
from ..model.solvers import SOLVERS


class FlowDelegate(DefinesCompletionDelegate):

    attribute = None

    def __init__(self, document, parent):
        super(FlowDelegate, self).__init__(document.defines.model, parent)
        self.solvers_model = document.solvers.model
        self.document = document

    def createEditor(self, parent, option, index):
        items = self.get_slots()
        if not items: return super(FlowDelegate, self).createEditor(parent, option, index)
        combo = QComboBox(parent)
        combo.setEditable(True)
        combo.addItems(items)
        combo.setEditText(index.data())
        try: combo.setCurrentIndex(items.index(index.data()))
        except ValueError: pass
        combo.setMaxVisibleItems(len(items))
        #self.connect(combo, SIGNAL("currentIndexChanged(int)"),
        #             self, SLOT("currentIndexChanged()"))
        #combo.currentIndexChanged[str].connect()
        return combo

    def get_slots(self):
        items = []
        for solver in self.solvers_model.entries:
            try:
                key = (solver.category, solver.solver)
                if key in self.document.solvers.model.local_solvers:
                    factory = self.document.solvers.model.local_solvers[key]
                else:
                    factory = SOLVERS.get(key)
            except AttributeError:
                pass
            else:
                if factory is not None:
                    items.extend(u'{}.{}'.format(solver.name, a[0]) for a in getattr(factory, self.attribute))
        return items


class ProvidersDelegate(FlowDelegate):
    attribute = 'providers'


class ReceiversDelegate(FlowDelegate):
    attribute = 'receivers'


class ConnectsController(TableController):

    def __init__(self, document, model=None):
        if model is None: model = ConnectsModel()
        TableController.__init__(self, document, model)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setItemDelegateForColumn(0, ReceiversDelegate(self.document, self.table))
        self.table.setItemDelegateForColumn(1, ProvidersDelegate(self.document, self.table))
