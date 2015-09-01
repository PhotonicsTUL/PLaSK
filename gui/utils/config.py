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

import sys
import os
from collections import OrderedDict
from ..qt import QtCore, QtGui

from numpy import log10, ceil

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
    'editor/matching_bracket_color': '#aaffaa',
    'editor/not_matching_bracket_color': '#ffaaaa',
    'launcher_local/font_size': 10,
    'launcher_local/font_bold': False,
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
    'geometry/selected_color': '#ff4444',
    'geometry/selected_alpha': 0.9,
    'geometry/selected_width': 3.0,
    'geometry/show_origin': True,
    'geometry/origin_color': '#ff4444',
    'geometry/origin_alpha': 0.9,
    'geometry/origin_width': 3.0,
    'geometry/origin_size': 10.0,
    'geometry/extra_color': '#00aaff',
    'geometry/extra_alpha': 0.9,
    'geometry/extra_width': 1.0,
}

if sys.platform == 'win32': DEFAULTS['editor/font_family'] = "Consolas"
elif sys.platform == 'darwin': DEFAULTS['editor/font_family'] = "Monaco"
else: DEFAULTS['editor/font_family'] = "Monospace"

DEFAULTS['launcher_local/font_family'] = DEFAULTS['editor/font_family']


def CheckBox(entry, help=None):
    return lambda parent: ConfigDialog.CheckBox(entry, help=help, parent=parent)

def Combo(entry, options, help=None):
    return lambda parent: ConfigDialog.Combo(entry, options, help=help, parent=parent)

def SpinBox(entry, min=None, max=None, help=None):
    return lambda parent: ConfigDialog.SpinBox(entry, min=min, max=max, help=help, parent=parent)

def FloatSpinBox(entry, step=None, min=None, max=None, help=None):
    return lambda parent: ConfigDialog.FloatSpinBox(entry, step=step, min=min, max=max, help=help, parent=parent)

def Color(entry, help=None):
    return lambda parent: ConfigDialog.Color(entry, help=help, parent=parent)

def Syntax(entry, help=None):
    return lambda parent: ConfigDialog.Syntax(entry, help=help, parent=parent)

def Font(family_entry, size_entry, bold_entry=None, style_entry=None, help=None):
    return lambda parent: ConfigDialog.Font(family_entry, size_entry, bold_entry, style_entry,
                                            help=help, parent=parent)


CONFIG_WIDGETS = OrderedDict([
    ("General Settings", [
        ("Create backup files on save",
         CheckBox('main_window/make_backup',
                  "Create backup files on save. "
                  "It is recommended to keep this option on, to keep the backup of the "
                  "edited files in case the new one becomes corrupt or you accidentally "
                  "remove some important parts.")),
        ("Icons theme (requires restart)",
         Combo('main_window/icons_theme',
               ['Tango', 'Breeze'] if os.name == 'nt' else ['system', 'Tango', 'Breeze'],
               "Main window icons theme.")),
    ]),
    ("Window Display", [
        "Geometry View",
        ("Selection frame color", Color('geometry/selected_color',
                                        "Color of a frame around the selected object.")),
        ("Selection frame opacity", FloatSpinBox('geometry/selected_alpha',
                                                 step=0.1, min=0.0, max=1.0,
                                                 help="Opacity of a frame around "
                                                      "the selected object.")),
        ("Selection frame width", FloatSpinBox('geometry/selected_width',
                                               step=0.1, min=0.1,
                                               help="Width of a frame around "
                                                    "the selected object.")),
        ("Show local origin", CheckBox('geometry/show_origin',
                                       "Show local origin of the selected object")),
        ("Origin mark color", Color('geometry/origin_color',
                                    "Color of a local origin mark.")),
        ("Origin mark opacity", FloatSpinBox('geometry/origin_alpha',
                                             step=0.1, min=0.0, max=1.0,
                                             help="Opacity of a local origin mark.")),
        ("Origin mark size", FloatSpinBox('geometry/origin_size',
                                          step=0.1, min=0.1,
                                          help="Size of a local origin mark.")),
        ("Origin mark width", FloatSpinBox('geometry/origin_width',
                                           step=0.1, min=0.1,
                                           help="Line width of the local origin mark.")),
        ("Info lines color", Color('geometry/extra_color',
                                   "Color of info lines for the selected object.")),
        ("Info lines opacity", FloatSpinBox('geometry/extra_alpha',
                                            step=0.1, min=0.0, max=1.0,
                                            help="Opacity of info lines for the selected object.")),
        ("Info lines width", FloatSpinBox('geometry/extra_width',
                                          step=0.1, min=0.1,
                                          help="Width of info lines for the selected object.")),

        "Text Editor",
        ("Editor font", Font('editor/font_family', 'editor/font_size',
                              help="Font size in text editors.")),
        ("Current line color", Color('editor/current_line_color',
                                     "Background color of the current line.")),
        ("Find result color", Color('editor/match_color',
                                    "Background color of strings matching current search.")),
        ("Replaced result color", Color('editor/replace_color',
                                        "Background color of strings right after replace.")),
        ("Word highlight color", Color('editor/selection_color',
                                       "Highlight color for the current word.")),
        ("Matching bracket color", Color('editor/matching_bracket_color',
                                         "Highlight color for matching brackets "
                                         "in script editor.")),
        ("Unmatched bracket color", Color('editor/not_matching_bracket_color',
                                          "Highlight color for unmatched brackets "
                                          "in script editor.")),
        "Launcher",
        # ("Font size", SpinBox('launcher_local/font_size', min=1,
        #                       help="Font size in local launcher window.")),
        ("Messages font", Font('launcher_local/font_family', 'launcher_local/font_size', 'launcher_local/font_bold',
                               help="Font in local launcher window.")),

    ]),
    ("Syntax Highlighting", [
        "Python Syntax",
        ("Comment", Syntax('syntax/python_comment', "Python syntax highlighting.")),
        ("String", Syntax('syntax/python_string', "Python syntax highlighting.")),
        ("Builtin", Syntax('syntax/python_builtin', "Python syntax highlighting.")),
        ("Keyword", Syntax('syntax/python_keyword', "Python syntax highlighting.")),
        ("Number", Syntax('syntax/python_number', "Python syntax highlighting.")),
        ("Class member", Syntax('syntax/python_member', "Python syntax highlighting.")),
        ("PLaSK function", Syntax('syntax/python_plask', "Python syntax highlighting.")),
        ("PLaSK provider", Syntax('syntax/python_provider', "Python syntax highlighting.")),
        ("PLaSK receiver", Syntax('syntax/python_receiver', "Python syntax highlighting.")),
        ("Log level", Syntax('syntax/python_log', "Python syntax highlighting.")),
        ("Solver", Syntax('syntax/python_solver', "Python syntax highlighting.")),
        ("XPL Definition", Syntax('syntax/python_define', "Python syntax highlighting.")),
        ("PLaSK dictionary", Syntax('syntax/python_loaded', "Python syntax highlighting.")),
        ("Pylab identifier", Syntax('syntax/python_pylab', "Python syntax highlighting.")),
        "XML Syntax",
        ("XML Tag", Syntax('syntax/xml_tag', "XML syntax highlighting.")),
        ("XML Attribute", Syntax('syntax/xml_attr', "XML syntax highlighting.")),
        ("XML Value", Syntax('syntax/xml_value', "XML syntax highlighting.")),
        ("XML Text", Syntax('syntax/xml_text', "XML syntax highlighting.")),
        ("XML Comment", Syntax('syntax/xml_comment', "XML syntax highlighting.")),
    ]),
])


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
            if help is not None: self.setWhatsThis(help)
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
        def __init__(self, entry, parent=None, min=None, max=None, help=None):
            super(ConfigDialog.SpinBox, self).__init__(parent)
            self.entry = entry
            if min is not None: self.setMinimum(min)
            if max is not None: self.setMaximum(min)
            self.setValue(int(CONFIG[entry]))
            if help is not None: self.setWhatsThis(help)
        def save(self):
            CONFIG[self.entry] = self.value()

    class FloatSpinBox(QtGui.QDoubleSpinBox):
        def __init__(self, entry, parent=None, step=None, min=None, max=None, help=None):
            super(ConfigDialog.FloatSpinBox, self).__init__(parent)
            self.entry = entry
            if min is not None: self.setMinimum(min)
            if max is not None: self.setMaximum(max)
            if step is not None:
                self.setSingleStep(step)
                self.setDecimals(int(ceil(-log10(step))))
            self.setValue(float(CONFIG[entry]))
            if help is not None: self.setWhatsThis(help)
        def save(self):
            CONFIG[self.entry] = self.value()

    class Color(QtGui.QToolButton):
        def __init__(self, entry, parent=None, help=None):
            super(ConfigDialog.Color, self).__init__(parent)
            self.entry = entry
            self._color = CONFIG[entry]
            self.setStyleSheet(u"background-color: {};".format(self._color))
            if help is not None: self.setWhatsThis(help)
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
            if help is not None: self.setWhatsThis(help)
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

    class Font(QtGui.QPushButton):
        def __init__(self, family_entry, size_entry, bold_entry=None, style_entry=None, parent=None, help=None):
            super(ConfigDialog.Font, self).__init__(parent)
            self.family_entry = family_entry
            self.style_entry = style_entry
            self.bold_entry = bold_entry
            self.size_entry = size_entry
            self.current_font = self.font()
            family = CONFIG[self.family_entry]
            if family is not None: self.current_font.setFamily(family)
            else: family = self.current_font.family()
            self.current_font.setBold(False)
            if bold_entry is not None:
                bold = CONFIG[bold_entry]
                if bold is not None: self.current_font.setBold(bold)
            self.current_font.setStyle(QtGui.QFont.StyleNormal)
            if style_entry is not None:
                style = CONFIG[style_entry]
                if style is not None: self.current_font.setStyle(style)
            size = CONFIG[self.size_entry]
            if size is not None: self.current_font.setPointSize(int(size))
            else: size = self.current_font.pointSize()
            # self.setFont(self.current_font)
            self.setText("{} {}".format(family, size))
            if help is not None:
                self.setWhatsThis(help)
            self.clicked.connect(self.on_press)
            self.setSizePolicy(QtGui.QSizePolicy.Expanding, self.sizePolicy().verticalPolicy())
        def on_press(self):
            dlg = QtGui.QFontDialog(self.parent())
            dlg.setCurrentFont(self.current_font)
            if dlg.exec_():
                self.current_font = dlg.selectedFont()
                self.setText("{} {}".format(self.current_font.family(), self.current_font.pointSize()))
        def save(self):
            CONFIG[self.family_entry] = self.current_font.family()
            CONFIG[self.size_entry] = self.current_font.pointSize()
            if self.bold_entry is not None:
                CONFIG[self.bold_entry] = self.current_font.bold()
            if self.style_entry is not None:
                CONFIG[self.style_entry] = self.current_font.style()

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

        self.items = []

        for cat, items in CONFIG_WIDGETS.items():
            tab = QtGui.QGroupBox(self)
            tab_layout = QtGui.QFormLayout()
            tab.setLayout(tab_layout)
            stack.addWidget(tab)
            categories.addItem(cat)
            hr = ""
            for item in items:
                if isinstance(item, str):
                    tab_layout.addRow(QtGui.QLabel(hr+"<b>"+item+"</b>", self))
                    hr = "<hr/>"
                else:
                    widget = item[1](self)
                    self.items.append(widget)
                    tab_layout.addRow(item[0], widget)

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
        for item in self.items:
            item.save()
        CONFIG.sync()
        from .widgets import EDITOR_FONT
        EDITOR_FONT.setFamily(CONFIG['editor/font_family'])
        EDITOR_FONT.setPointSize(int(CONFIG['editor/font_size']))
        self.parent().config_changed.emit()

    def accept(self):
        self.apply()
        super(ConfigDialog, self).accept()

