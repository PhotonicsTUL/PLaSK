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

from ..model.connects import ConnectsModel
from .table import TableController
from .defines import DefinesCompletionDelegate

class ConnectsController(TableController):

    def __init__(self, document, model=None):
        if model is None: model = ConnectsModel()
        TableController.__init__(self, document, model)
        for i in range(0, 2):
            self.table.setItemDelegateForColumn(i, DefinesCompletionDelegate(self.document.defines.model, self.table))
