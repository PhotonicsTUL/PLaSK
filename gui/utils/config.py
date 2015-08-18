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
        if not item: continue
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

    class Color(QtGui.QToolButton):
        def __init__(self, entry, parent=None, help=None):
            super(ConfigDialog.Color, self).__init__(parent)
            self.entry = entry
            self._color = CONFIG[entry]
            self.setStyleSheet(u"background-color: {};".format(self._color))
            if help is not None:
                self.setWhatsThis(help)
            self.clicked.connect(self.on_press)
            self.setSizePolicy(QtGui.QSizePolicy.Expanding, self.sizePolicy().verticalPolicy())
        def on_press(self):
            dlg = QtGui.QColorDialog(self.parent())
            if self._color:
                dlg.setCurrentColor(QtGui.QColor(self._color))
            if dlg.exec_():
                self._color = dlg.currentColor().name()
                self.setStyleSheet(u"background-color: {};".format(self._color))
        def save(self):
            CONFIG[self.entry] = self._color

    class Syntax(QtGui.QWidget):
        def __init__(self, entry, parent=None, help=None):
            super(ConfigDialog.Syntax, self).__init__(parent)
            self.entry = entry
            syntax = parse_highlight(CONFIG[entry])
            layout = QtGui.QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            self.setLayout(layout)
            self.color_button = QtGui.QToolButton(self)
            self.color_button.setSizePolicy(QtGui.QSizePolicy.Expanding, self.color_button.sizePolicy().verticalPolicy())
            self.color_button.clicked.connect(self.on_color_press)
            layout.addWidget(self.color_button)
            self._color = syntax.get('color')
            if self._color is not None:
                self.color_button.setStyleSheet(u"background-color: {};".format(self._color))
            layout.addWidget(self.color_button)
            self.bold = QtGui.QCheckBox('bold', self)
            self.bold.setChecked(syntax.get('bold', False))
            layout.addWidget(self.bold)
            self.italic = QtGui.QCheckBox('italic', self)
            self.italic.setChecked(syntax.get('italic', False))
            layout.addWidget(self.italic)
            if help is not None:
                self.setWhatsThis(help)
        def on_color_press(self):
            if QtGui.QApplication.keyboardModifiers() == QtCore.Qt.CTRL:
                self._color = None
                self.color_button.setStyleSheet("")
                return
            dlg = QtGui.QColorDialog(self.parent())
            if self._color:
                dlg.setCurrentColor(QtGui.QColor(self._color))
            if dlg.exec_():
                self._color = dlg.currentColor().name()
                self.color_button.setStyleSheet(u"background-color: {};".format(self._color))
        def save(self):
            syntax = []
            if self._color is not None: syntax.append('color=' + self._color)
            if self.bold.isChecked(): syntax.append('bold=true')
            if self.italic.isChecked(): syntax.append('italic=true')
            CONFIG[self.entry] = ', '.join(syntax)

    def __init__(self, parent):
        super(ConfigDialog, self).__init__(parent)
        self.setWindowTitle("GUI Settings")
        vlayout = QtGui.QVBoxLayout()

        categories = QtGui.QListWidget()
        stack = QtGui.QStackedWidget()
        categories.currentRowChanged.connect(stack.setCurrentIndex)

        hlayout = QtGui.QHBoxLayout()
        hlayout.addWidget(categories)
        hlayout.addWidget(stack)
        vlayout.addLayout(hlayout)

        # current_layout = QtGui.QFormLayout()

        self.items = [
            ("General", (
                ("Create backup files on save",
                 ConfigDialog.CheckBox('main_window/make_backup', self,
                                       "Create backup files on save. "
                                       "It is recommended to keep this option on, to keep the backup of the "
                                       "edited files in case the new one becomes corrupt or you accidentally "
                                       "remove some important parts.")),
                ("Icons theme (requires restart)",
                 ConfigDialog.Combo('main_window/icons_theme',
                                    ['Tango', 'Breeze'] if os.name == 'nt' else ['system', 'Tango', 'Breeze'],
                                    self,
                                    "Main window icons theme.")),
            )),
            ("Text Editor", (
                ("Font size",
                 ConfigDialog.SpinBox('editor/font_size', self,
                                      "Font size in text editors.")),
                ("Find result color", ConfigDialog.Color('editor/match_color', self,
                                                         "Background color of strings matching current search.")),
                ("Replaced result color", ConfigDialog.Color('editor/replace_color', self,
                                                             "Background color of strings right after replace.")),
                ("Word highlight color", ConfigDialog.Color('editor/selection_color', self,
                                                            "Highlight color for the current word.")),
            )),
            ("Syntax Highlighting", (
                "Python Syntax",
                ("Comment", ConfigDialog.Syntax('syntax/python_comment', self, "Python syntax highlighting.")),
                ("String", ConfigDialog.Syntax('syntax/python_string', self, "Python syntax highlighting.")),
                ("Builtin", ConfigDialog.Syntax('syntax/python_builtin', self, "Python syntax highlighting.")),
                ("Keyword", ConfigDialog.Syntax('syntax/python_keyword', self, "Python syntax highlighting.")),
                ("Number", ConfigDialog.Syntax('syntax/python_number', self, "Python syntax highlighting.")),
                ("Class member", ConfigDialog.Syntax('syntax/python_member', self, "Python syntax highlighting.")),
                ("PLaSK function", ConfigDialog.Syntax('syntax/python_plask', self, "Python syntax highlighting.")),
                ("PLaSK provider", ConfigDialog.Syntax('syntax/python_provider', self, "Python syntax highlighting.")),
                ("PLaSK receiver", ConfigDialog.Syntax('syntax/python_receiver', self, "Python syntax highlighting.")),
                ("Log level", ConfigDialog.Syntax('syntax/python_log', self, "Python syntax highlighting.")),
                ("Solver", ConfigDialog.Syntax('syntax/python_solver', self, "Python syntax highlighting.")),
                ("XPL Definition", ConfigDialog.Syntax('syntax/python_define', self, "Python syntax highlighting.")),
                ("PLaSK dictionary", ConfigDialog.Syntax('syntax/python_loaded', self, "Python syntax highlighting.")),
                ("Pylab identifier", ConfigDialog.Syntax('syntax/python_pylab', self, "Python syntax highlighting.")),
                "<hr/>XML Syntax",
                ("XML Tag", ConfigDialog.Syntax('syntax/xml_tag', self, "XML syntax highlighting.")),
                ("XML Attribute", ConfigDialog.Syntax('syntax/xml_attr', self, "XML syntax highlighting.")),
                ("XML Value", ConfigDialog.Syntax('syntax/xml_value', self, "XML syntax highlighting.")),
                ("XML Text", ConfigDialog.Syntax('syntax/xml_text', self, "XML syntax highlighting.")),
                ("XML Comment", ConfigDialog.Syntax('syntax/xml_comment', self, "XML syntax highlighting.")),
            )),
        ]

        for cat, items in self.items:
            tab = QtGui.QGroupBox(self)
            tab_layout = QtGui.QFormLayout()
            tab.setLayout(tab_layout)
            stack.addWidget(tab)
            categories.addItem(cat)
            for item in items:
                if isinstance(item, str):
                    tab_layout.addRow(QtGui.QLabel("<b>"+item+"</b>", self))
                else:
                    tab_layout.addRow(*item)

        categories.setFixedWidth(categories.sizeHintForColumn(0) + 4)

        buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Apply | QtGui. QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QtGui.QDialogButtonBox.Apply).clicked.connect(self.apply)

        vlayout.addWidget(buttons)
        self.setLayout(vlayout)

        self.resize(600, 0)

    def apply(self):
        for _, items in self.items:
            for item in items:
                if isinstance(item, tuple):
                    item[1].save()
        CONFIG.sync()
        from .widgets import DEFAULT_FONT
        DEFAULT_FONT.setPointSize(int(CONFIG['editor/font_size']))
        self.parent().config_changed.emit()

    def accept(self):
        self.apply()
        super(ConfigDialog, self).accept()

