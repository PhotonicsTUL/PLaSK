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

from ...qt import QtCore
from ...qt.QtCore import Qt

try:
    import jedi
except ImportError:
    jedi = None


class CompletionsModel(QtCore.QAbstractTableModel):

    # _icon_map = {}

    def __init__(self, items):
        super(CompletionsModel, self).__init__()
        self.items = items

        # if not self._icon_map:
        #     self._load_icons()
        #
    # def _load_icons(self):
    #     func_icon = QIcon.fromTheme("code-function")
    #     var_icon = QIcon.fromTheme("code-variable")
    #     class_icon = QIcon.fromTheme("code-class")
    #     builtin_icon = QIcon.fromTheme("code-typedef")
    #     imported_icon = QIcon.fromTheme("code-block")
    #     instance_icon = class_icon
    #     module_icon = QIcon.fromTheme("code-context")
    #     self._icon_map.update({
    #                           None: QIcon(),
    #                           "function": func_icon,
    #                           "variable": var_icon,
    #                           "parameter": var_icon,
    #                           "class": class_icon,
    #                           "builtin": builtin_icon,
    #                           "imported": imported_icon,
    #                           "instance": instance_icon,
    #                           "module": module_icon,
    #                           })

    # def headerData(self, section, orientation, role=Qt.DisplayRole):
    #    return self.columns[section]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role in (Qt.DisplayRole, Qt.EditRole, Qt.DecorationRole):
            row = index.row()
            col = index.column()
            return self.items[row].name
            # if col == 0:
            #     value = self._icon_map[value]
        else:
            return None

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.items)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 1


def get_completions(text, block, column):
    if jedi is None: return
    script = jedi.Script(text, block+1, column)
    return script.completions()