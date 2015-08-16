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

import os
from ..qt import QtCore, QtGui

_parsed = {'true': True, 'yes': True, 'false': False, 'no': False}


DEFAULTS = {
    'main_window/make_backup': True,
    'main_window/use_menu': False,
    'main_window/icons_theme': 'system',
    'editor/font_size': QtGui.QFont().pointSize(),
    'editor/current_line_color': '#ffffee',
    'editor/selection_color': '#ffffdd',
    'editor/match_color': '#ddffdd',
    'editor/replace_color': '#ffddff',
    'launcher_local/font_size': 10,
    'syntax/xml_comment': 'color=green, italic=true',
    'syntax/xml_tag': 'color=maroon, bold=true',
    'syntax/xml_attr': 'color=#888800',
    'syntax/xml_value': 'color=darkblue',
    'syntax/xml_text': 'color=black',
    'syntax/python_comment': 'color=green, italic=true',
    'syntax/python_string': 'color=blue',
    'syntax/python_builtin': 'color=maroon',
    'syntax/python_keyword': 'color=black, bold=true',
    'syntax/python_number': 'color=darkblue',
    'syntax/python_member': 'color=#444400',
    'syntax/python_plask': 'color=#0088ff',
    'syntax/python_provider': 'color=#888800',
    'syntax/python_receiver': 'color=#888800',
    'syntax/python_log': 'color=blue',
    'syntax/python_solver': 'color=red',
    'syntax/python_define': 'italic=true',
    'syntax/python_loaded': 'color=#ff8800',
    'syntax/python_pylab': 'color=#440088',
    'editor/matching_bracket_color': '#aaffaa',
    'editor/not_matching_bracket_color': '#ffaaaa',
}


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

    def __getitem__(self, key):
        current = self.qsettings.value(key)
        if current is None:
            return DEFAULTS.get(key)
        try:
            return _parsed.get(current, current)
        except TypeError:
            return current

    def get(self, key, default):
        current = self.qsettings.value(key)
        if current is None:
            return default
        try:
            return _parsed.get(current, current)
        except TypeError:
            return current

    def __setitem__(self, key, value):
        self.qsettings.setValue(key, value)

    def sync(self):
        """Synchronize settings"""
        self.qsettings.sync()

CONFIG = Config()


class ConfigDialog(QtGui.QDialog):

    class CheckBox(QtGui.QCheckBox):
        def __init__(self, entry, parent=None, help=None):
            super(ConfigDialog.CheckBox, self).__init__(parent)
            self.entry = entry
            self.setChecked(bool(CONFIG[entry]))
            if help is not None:
                self.setWhatsThis(help)
        def save(self):
            CONFIG[self.entry] = self.isChecked()

    class Combo(QtGui.QComboBox):
        def __init__(self, entry, options, parent=None, help=None):
            super(ConfigDialog.Combo, self).__init__(parent)
            self.entry = entry
            self.addItems(options)
            try:
                index = options.index(CONFIG[entry])
            except ValueError:
                index = 0
            self.setCurrentIndex(index)
            if help is not None:
                self.setWhatsThis(help)
        def save(self):
            CONFIG[self.entry] = self.currentText()

    class SpinBox(QtGui.QSpinBox):
        def __init__(self, entry, parent=None, help=None):
            super(ConfigDialog.SpinBox, self).__init__(parent)
            self.entry = entry
            self.setMinimum(1)
            self.setValue(int(CONFIG[entry]))
            if help is not None:
                self.setWhatsThis(help)
        def save(self):
            CONFIG[self.entry] = self.value()

    class Color(QtGui.QPushButton):
        pass

    def __init__(self, parent):
        super(ConfigDialog, self).__init__(parent)
        self.setWindowTitle("GUI Settings")
        vlayout = QtGui.QVBoxLayout()

        groups = QtGui.QListWidget()
        stack = QtGui.QStackedWidget()
        groups.currentRowChanged.connect(stack.setCurrentIndex)

        hlayout = QtGui.QHBoxLayout()
        hlayout.addWidget(groups)
        hlayout.addWidget(stack)
        vlayout.addLayout(hlayout)

        # current_layout = QtGui.QFormLayout()

        self.items = [
            "General",
            ("Create backup files on save",
             ConfigDialog.CheckBox('main_window/make_backup', self,
                                   "Create backup files on save.\n\n"
                                   "It is recommended to keep this option on, to keep the backup of the "
                                   "edited files in case the new one becomes corrupt or you accidentally "
                                   "remove some important parts.")),
            ("Show menu bar (requires restart)",
             ConfigDialog.CheckBox('main_window/use_menu', self,
                                   "Show menu bar.\n\n"
                                   "Setting this option can be useful in Ubuntu Linux, so the available "
                                   "operations appear in the global menu and are accessible through HUD. "
                                   "If you are not using Ubuntu, you probably may leave it unchecked.")),
            ("Icons theme (requires restart)",
             ConfigDialog.Combo('main_window/icons_theme', ['system', 'Tango', 'Breeze'], self,
                                "Main window icons theme.")),
            "Text Editor",
            ("Font size",
             ConfigDialog.SpinBox('editor/font_size', self,
                                  "Font size in text editors."))

        ]

        for item in self.items:
            if isinstance(item, str):
                tab = QtGui.QGroupBox(item)
                current_layout = QtGui.QFormLayout()
                tab.setLayout(current_layout)
                stack.addWidget(tab)
                groups.addItem(item)
            else:
                current_layout.addRow(*item)

        groups.setFixedWidth(groups.sizeHintForColumn(0) + 4)

        buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Apply | QtGui. QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QtGui.QDialogButtonBox.Apply).clicked.connect(self.apply)

        vlayout.addWidget(buttons)
        self.setLayout(vlayout)

    def apply(self):
        for item in self.items:
            if not isinstance(item, str):
                item[1].save()
        CONFIG.sync()
        self.parent().config_changed.emit()

    def accept(self):
        self.apply()
        super(ConfigDialog, self).accept()

