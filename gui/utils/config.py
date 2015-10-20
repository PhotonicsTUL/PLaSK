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

if sys.platform == 'win32': _default_font_family = "Consolas"
elif sys.platform == 'darwin': _default_font_family = "Monaco"
else:_default_font_family = "Monospace"


DEFAULTS = {
    'main_window/make_backup': True,
    'main_window/use_menu': False,
    'main_window/icons_theme': 'system',
    'editor/font': [_default_font_family, str(QtGui.QFont().pointSize()),
                    '-1', '5', '50', '0', '0', '0', '0', '0'],
    'editor/help_font': [_default_font_family, str(QtGui.QFont().pointSize()-2),
                         '-1', '5', '50', '0', '0', '0', '0', '0'],
    'editor/current_line_color': '#ffffee',
    'editor/selection_color': '#ffffdd',
    'editor/match_color': '#ddffdd',
    'editor/replace_color': '#ffddff',
    'editor/matching_bracket_color': '#aaffaa',
    'editor/not_matching_bracket_color': '#ffaaaa',
    'launcher_local/font': [_default_font_family, str(QtGui.QFont().pointSize()-1),
                            '-1', '5', '50', '0', '0', '0', '0', '0'],
    'syntax/xml_comment': 'color=green, italic=true',
    'syntax/xml_tag': 'color=maroon, bold=true',
    'syntax/xml_attr': 'color=#888800',
    'syntax/xml_value': 'color=darkblue',
    'syntax/xml_text': 'color=black',
    'syntax/python_comment': 'color=#008000, italic=true',
    'syntax/python_string': 'color=#0000ff',
    'syntax/python_keyword': 'color=black, bold=true',
    'syntax/python_number': 'color=darkblue',
    'syntax/python_builtin': 'color=maroon',
    'syntax/python_member': 'color=#444400',
    'syntax/python_plask': 'color=#aa007f',
    'syntax/python_provider': 'color=#888800',
    'syntax/python_receiver': 'color=#888800',
    'syntax/python_log': 'color=blue',
    'syntax/python_solver': 'color=red',
    'syntax/python_loaded': 'color=#ff8800',
    'syntax/python_pylab': 'color=#440088',
    'syntax/python_define': 'color=#1c68b9, italic=true',
    'geometry/selected_color': '#ff4444',
    'geometry/selected_alpha': 0.7,
    'geometry/selected_width': 2.0,
    'geometry/show_origin': True,
    'geometry/origin_color': '#ffea00',
    'geometry/origin_alpha': 0.7,
    'geometry/origin_width': 2.0,
    'geometry/origin_size': 10.0,
    'geometry/extra_color': '#00aaff',
    'geometry/extra_alpha': 0.7,
    'geometry/extra_width': 1.0,
    'mesh/mesh_color': '#00aa00',
    'mesh/line_width': 1.0,
    'workarounds/jedi_no_dot': False,
    'workarounds/no_jedi': False,
    'workarounds/blocking_jedi': False,
}


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

def Font(entry, help=None):
    return lambda parent: ConfigDialog.Font(entry, help=help, parent=parent)


CONFIG_WIDGETS = OrderedDict([
    ("General Settings", [
        "General Settings",
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

        "Mesh Preview",
        ("Mesh color", Color('mesh/mesh_color', "Mesh color in the preview plot.")),
        ("Mesh line width", FloatSpinBox('mesh/line_width', step=0.1, min=0.1,
                                         help="Mesh lines width in the preview plot.")),

        "Text Editor",
        ("Editor font", Font('editor/font', "Font in text editors.")),
        ("Help font", Font('editor/help_font', "Font in script on-line help.")),
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
        ("Messages font", Font('launcher_local/font', "Font in local launcher window.")),

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
    ("Workarounds", [
        "Script Completion",
        ("Do not complete on dot", CheckBox('workarounds/jedi_no_dot',
                                            "Do not show completion pop-up after you type a dot. This still allows "
                                            "to show the pop-up manually by pressing Ctrl+Space.")),
        ("Run in a single thread", CheckBox('workarounds/blocking_jedi',
                                            "Do not run script completion in background. This may cause the GUI stop "
                                            "responding for the couple of seconds when showing completion pop-up, but "
                                            "may be helpful if the program often crashes on completion.")),
        ("Disable completion", CheckBox('workarounds/no_jedi',
                                        "Disable script completion and on-line help.")),
    ])
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

    def __delitem__(self, key):
        self.qsettings.remove(key)

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
        def __init__(self, entry, parent=None, help=None):
            super(ConfigDialog.Font, self).__init__(parent)
            self.entry = entry
            self.current_font = QtGui.QFont()
            self.current_font.fromString(','.join(CONFIG[entry]))
            family = self.current_font.family()
            size = self.current_font.pointSize()
            self.setText("{} {}".format(family, size))
            self.setFont(self.current_font)
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
                self.setFont(self.current_font)
        def save(self):
            CONFIG[self.entry] = self.current_font.toString().split(',')

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

        self.items = []

        for cat, items in CONFIG_WIDGETS.items():
            # page = QtGui.QToolBox()
            page = QtGui.QTabWidget()
            stack.addWidget(page)
            categories.addItem(cat)
            tab = None
            for item in items:
                if isinstance(item, str):
                    tab = QtGui.QWidget(page)
                    tab_layout = QtGui.QFormLayout()
                    tab.setLayout(tab_layout)
                    # page.addItem(tab, item)
                    page.addTab(tab, item)
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

        self.resize(600, 600)

    def apply(self):
        for item in self.items:
            item.save()
        CONFIG.sync()
        from .widgets import EDITOR_FONT
        EDITOR_FONT.fromString(','.join(CONFIG['editor/font']))
        self.parent().config_changed.emit()

    def accept(self):
        self.apply()
        super(ConfigDialog, self).accept()

